"""All paid calls live here. Cache hits and replay never construct an API client."""

import asyncio
import base64
import hashlib
import io
import json
import math
import os
from collections import deque
from contextlib import aclosing
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from tempfile import NamedTemporaryFile

from openai import APIConnectionError, APIStatusError, AsyncOpenAI, OpenAI
from PIL import Image
from tenacity import AsyncRetrying, Retrying, retry_if_exception, stop_after_attempt, wait_exponential

BASE_URL = "https://api.meta.ai/v1"
SPARK = "muse-spark-1.3"
WONDER = "muse-wonder-1.0"
SAM = "sam-3.1"
# USD per million tokens; SAM's tariff is USD per thousand input images.
PRICES = {
    SPARK: dict(input="1.25", cached_input="0.15", output="4.25"),
    WONDER: dict(input="4.00", cached_input="0.20", output="20.00"),
    SAM: dict(images="2.50"),
}
MAX_COMPLETION_TOKENS = 8192
REQUEST_TIMEOUT = 60


class CacheMiss(RuntimeError):
    pass


class SpendLimit(RuntimeError):
    pass


class ResponseError(ValueError):
    pass


class MissingKey(RuntimeError):
    pass


def load_api_key():
    """Credentials come only from the process environment (see AGENTS.md)."""
    key = os.environ.get("MODEL_API_KEY", "").strip()
    if key:
        return key
    raise MissingKey("MODEL_API_KEY missing; set it in the environment")


def usage_cost(model, usage, images=0):
    """Prompt usage includes cached tokens; do not bill that subset twice."""
    price = PRICES[model]
    if model == SAM:
        return Decimal(images) * Decimal(price["images"]) / 1000
    if not usage or usage.get("input_tokens") is None or usage.get("output_tokens") is None:
        return None
    cached = usage.get("cached_input_tokens", 0)
    return (Decimal(usage["input_tokens"] - cached) * Decimal(price["input"]) +
            Decimal(cached) * Decimal(price["cached_input"]) +
            Decimal(usage["output_tokens"]) * Decimal(price["output"])) / 1_000_000


def response_usage(raw):
    """Normalize Chat Completions and Responses usage (including reasoning output)."""
    usage = raw.get("usage") or {}
    incoming = usage.get("prompt_tokens", usage.get("input_tokens"))
    outgoing = usage.get("completion_tokens", usage.get("output_tokens"))
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    cached = details.get("cached_tokens")
    if cached is None:
        cached = 0
    for value in (incoming, outgoing, cached):
        if value is not None and (type(value) is not int or value < 0):
            raise ResponseError("Invalid response token usage")
    if incoming is not None and cached > incoming:
        raise ResponseError("Cached tokens exceed input tokens")
    return dict(input_tokens=incoming, cached_input_tokens=cached, output_tokens=outgoing)


def request_size(payload):
    """Count prompt bytes without base64, and estimate image tokens by 512px tiles.

    One token per UTF-8 byte plus framing is deliberately conservative for text.
    Image tokens are a reservation heuristic, not a claimed provider tokenizer.
    """
    images, image_tokens = 0, 0

    def clean(value):
        nonlocal images, image_tokens
        if isinstance(value, dict):
            if value.get("type") in {"image_url", "input_image"}:
                images += 1
                url = value.get("image_url", "")
                if isinstance(url, dict):
                    url = url.get("url", "")
                try:
                    with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as image:
                        tiles = math.ceil(image.width / 512) * math.ceil(image.height / 512)
                except (ValueError, IndexError, OSError):
                    tiles = 4
                image_tokens += 1024 * tiles
                return "[image]"
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    prompt_bytes = len(json.dumps(clean(payload), ensure_ascii=False).encode("utf-8"))
    return dict(input_tokens=prompt_bytes + 256 + image_tokens, cached_input_tokens=0,
                output_tokens=payload.get("max_completion_tokens", MAX_COMPLETION_TOKENS)), images


def retryable(exc):
    return isinstance(exc, (APIConnectionError, TimeoutError)) or (
        isinstance(exc, APIStatusError) and (exc.status_code == 429 or 500 <= exc.status_code < 600))


class SpendGuard:
    def __init__(self, cap=3):
        self.cap = Decimal(str(cap))
        if not self.cap.is_finite() or self.cap < 0:
            raise ValueError("Spend cap must be finite and nonnegative")
        self.reserved = Decimal(0)
        self.calls = 0

    def reserve(self, amount):
        amount = Decimal(str(amount))
        if not amount.is_finite() or amount <= 0:
            raise ValueError("Call reservation must be finite and positive")
        if self.reserved + amount > self.cap:
            raise SpendLimit(f"Call would exceed USD {self.cap} run cap (reserved {self.reserved})")
        # Failed requests may have been billed: never refund their reservation.
        self.reserved += amount
        self.calls += 1
        return amount

    def reconcile(self, reservation, actual):
        if actual is not None:
            self.reserved += actual - reservation

    def summary(self):
        return dict(cap_usd=float(self.cap), reserved_usd=float(self.reserved), attempted_calls=self.calls)


def image_data_url(image):
    return "data:image/png;base64," + base64.b64encode(image).decode("ascii")


def request_hash(namespace, operation, payload):
    body = json.dumps([namespace, operation, payload], sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode()).hexdigest()


async def sam_events(stream):
    """Cookbook adapter: close output lanes also on content_part.done."""
    completed = set()
    async for event in stream:
        data = event.model_dump() if hasattr(event, "model_dump") else dict(event)
        lane = tuple(data.get(k) for k in ("item_id", "output_index", "content_index"))
        if data.get("type") == "response.output_text.done":
            if lane not in completed:
                completed.add(lane)
                yield data
            continue
        yield data
        part = data.get("part")
        if (data.get("type") == "response.content_part.done" and lane not in completed
                and isinstance(part, dict) and part.get("type") == "output_text"
                and isinstance(part.get("text"), str)):
            completed.add(lane)
            yield dict(type="response.output_text.done", item_id=data.get("item_id"),
                       output_index=data.get("output_index"), content_index=data.get("content_index"),
                       text=part["text"])


async def parse_sam_events(events):
    from meta_sam_parser import ResponsesStreamError, image_segmentation_format, parse_responses_stream

    try:
        result = await parse_responses_stream(events, image_segmentation_format()).final_result()
    except ResponsesStreamError as exc:
        if exc.cause is not None and retryable(exc.cause):
            raise exc.cause
        raise ResponseError(f"SAM stream failed: {exc.code}") from exc
    if result.outcome.status != "completed" or any(d.severity == "error" for d in result.diagnostics):
        raise ResponseError("SAM returned an incomplete or invalid segmentation")
    return dict(boxes=[asdict(r) for r in result.records if r.kind == "box"],
                masks=[asdict(r) for r in result.records if r.kind == "mask"],
                diagnostics=[asdict(d) for d in result.diagnostics])


class MetaClient:
    namespace = "meta-v1"
    simulated = False

    def __init__(self, cache_dir="cache", *, replay=False, cap=3, timeout=REQUEST_TIMEOUT,
                 max_attempts=3, use_cache=True):
        self.cache_dir = Path(cache_dir)
        self.replay = replay
        if replay and not use_cache:
            raise ValueError("Replay requires the cache")
        self.use_cache = use_cache
        self.guard = SpendGuard(cap)
        if not math.isfinite(timeout) or timeout <= 0 or type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("Timeout and retry attempts must be positive")
        self.timeout, self.max_attempts = timeout, max_attempts
        self.cache_hits = 0
        self.usage = []

    @staticmethod
    def _key():
        return load_api_key()

    def estimate(self, payload):
        usage, images = request_size(payload)
        return usage_cost(payload["model"], usage, images)

    def count_input_tokens(self, payload):
        """Upper bound, not an exact tokenizer count; never use the tile estimate."""
        # Cookbook 01_api_fundamentals/08_long_context: Spark context = 1,048,576.
        # Local Meta pricing lists Wonder as 1M context. A whole context per image
        # bounds any accepted image, regardless of resolution or URL encoding.
        image_bound = 1_048_576
        images = 0

        def without_images(value):
            nonlocal images
            if isinstance(value, dict):
                if value.get("type") in {"image_url", "input_image"}:
                    images += 1
                    return "[image]"
                return {k: without_images(v) for k, v in value.items()}
            if isinstance(value, list):
                return [without_images(v) for v in value]
            return value

        # Byte-level BPE merges bytes, so text tokens cannot exceed UTF-8 bytes.
        # Include all JSON (tools, schemas, tool results too), plus generous fixed
        # framing for each message and the assistant generation prefix.
        text_bytes = len(json.dumps(without_images(payload), ensure_ascii=False).encode("utf-8"))
        return text_bytes + 256 * (len(payload.get("messages", [])) + 1) + images * image_bound

    def reservation(self, payload):
        if payload["model"] == SAM:
            return self.estimate(payload)
        count = self.count_input_tokens(payload)
        if type(count) is not int or count < 0:
            raise SpendLimit("Invalid model input-token count")
        return usage_cost(payload["model"], dict(input_tokens=count,
            cached_input_tokens=0, output_tokens=payload["max_completion_tokens"]))

    def _record(self, payload, raw, source, *, failed=False, reservation=Decimal(0)):
        model = payload["model"]
        _, images = request_size(payload)
        usage = response_usage(raw)
        cost = usage_cost(model, usage, images) if not failed and source != "fake" else None
        self.usage.append(dict(model=model, source=source, failed=failed, images=images,
                               **usage, response_cost_usd=float(cost) if cost is not None else None,
                               cost_usd=float(cost) if cost is not None and source == "live" else 0,
                               unresolved_usd=float(reservation) if source == "live" and cost is None else 0))
        if source == "live":
            self.guard.reconcile(reservation, cost)

    def usage_summary(self, start=0):
        fields = ("calls", "paid_calls", "cache_hits", "simulated_calls", "failed_calls", "input_tokens",
                  "cached_input_tokens", "output_tokens", "images", "cost_usd", "unresolved_usd",
                  "unreported_usage_calls")
        models = {}
        for event in self.usage[start:]:
            row = models.setdefault(event["model"], dict.fromkeys(fields, 0))
            row["calls"] += 1
            row["paid_calls"] += event["source"] == "live"
            row["cache_hits"] += event["source"] == "cache"
            row["simulated_calls"] += event["source"] == "fake"
            row["failed_calls"] += event["failed"]
            row["unreported_usage_calls"] += (event["source"] != "fake" and
                                               event["input_tokens"] is None)
            for field in ("input_tokens", "cached_input_tokens", "output_tokens", "images",
                          "cost_usd", "unresolved_usd"):
                row[field] += event[field] or 0
        total = {field: sum(row[field] for row in models.values()) for field in fields}
        for row in [total, *models.values()]:
            for field in ("cost_usd", "unresolved_usd"):
                row[field] = round(row[field], 12)
        return dict(**total, by_model=models)

    def _retry_options(self):
        return dict(stop=stop_after_attempt(self.max_attempts), retry=retry_if_exception(retryable),
                    wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)

    def _cached(self, operation, payload, send, validate=None):
        if not self.use_cache:
            result = send(payload)
            if self.simulated:
                self._record(payload, result, "fake")
            if validate is not None:
                validate(result)
            return result
        digest = request_hash(self.namespace, operation, payload)
        path = self.cache_dir / f"{digest}.json"
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("request_hash") != digest or cached.get("namespace") != self.namespace:
                raise ResponseError("Invalid cache envelope")
            try:
                if validate is not None:
                    validate(cached["response"])
            except ResponseError:
                if self.replay:
                    raise
                failures = self.cache_dir / "failures"
                failures.mkdir(parents=True, exist_ok=True)
                path.replace(failures / path.name)
            else:
                self.cache_hits += 1
                self._record(payload, cached["response"], "fake" if self.simulated else "cache")
                return cached["response"]
        if self.replay:
            raise CacheMiss(f"Replay cache miss: {operation} {digest}")
        result = send(payload)
        if self.simulated:
            self._record(payload, result, "fake")
        if validate is not None:
            try:
                validate(result)
            except ResponseError:
                failures = self.cache_dir / "failures"
                failures.mkdir(parents=True, exist_ok=True)
                with NamedTemporaryFile(mode="w", encoding="utf-8", dir=failures,
                                        suffix=".json", delete=False) as handle:
                    json.dump(dict(request_hash=digest, response=result), handle, ensure_ascii=False)
                raise
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        envelope = dict(namespace=self.namespace, request_hash=digest, response=result)
        # Atomic replacement: an interrupted write must not become a cache hit.
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.cache_dir,
                                suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            try:
                json.dump(envelope, handle, ensure_ascii=False, allow_nan=False)
                handle.flush()
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        temporary.replace(path)
        return result

    def _send_chat(self, payload):
        key = self._key()
        for attempt in Retrying(**self._retry_options()):
            with attempt:
                reservation = self.guard.reserve(self.reservation(payload))
                try:
                    # Each tenacity retry gets its own budget reservation.
                    with OpenAI(base_url=BASE_URL, api_key=key, max_retries=0, timeout=self.timeout) as api:
                        raw = api.chat.completions.create(**payload).model_dump(mode="json")
                    self._record(payload, raw, "live", reservation=reservation)
                except Exception:
                    self._record(payload, {}, "live", failed=True, reservation=reservation)
                    raise
                return raw

    async def _sam_once(self, payload, key):
        usage = {}

        async def observed(events):
            async for event in events:
                if event.get("type") == "response.completed":
                    usage.update(event.get("response", {}).get("usage") or {})
                yield event

        async with AsyncOpenAI(base_url=BASE_URL, api_key=key, max_retries=0, timeout=self.timeout) as api:
            stream = await api.responses.create(**payload)
            async with stream, aclosing(sam_events(stream)) as events:
                async with aclosing(observed(events)) as measured:
                    result = await parse_sam_events(measured)
        return dict(**result, usage=usage)

    async def _sam(self, payload):
        key = self._key()
        async for attempt in AsyncRetrying(**self._retry_options()):
            with attempt:
                reservation = self.guard.reserve(self.reservation(payload))
                try:
                    result = await self._sam_once(payload, key)
                    self._record(payload, result, "live", reservation=reservation)
                except Exception:
                    self._record(payload, {}, "live", failed=True, reservation=reservation)
                    raise
                return result

    def _send_sam(self, payload):
        return asyncio.run(self._sam(payload))

    def chat(self, prompt, *, image=None, schema=None, schema_name="result", model=SPARK,
             max_completion_tokens=MAX_COMPLETION_TOKENS):
        if type(max_completion_tokens) is not int or max_completion_tokens <= 0:
            raise ValueError("max_completion_tokens must be a positive integer")
        content = [{"type": "text", "text": prompt}]
        if image is not None:
            content.append({"type": "image_url", "image_url": {"url": image_data_url(image)}})
        payload = dict(model=model, messages=[dict(role="user", content=content)],
                       max_completion_tokens=max_completion_tokens)
        if schema is not None:
            payload["response_format"] = dict(type="json_schema", json_schema=dict(
                name=schema_name, strict=True, schema=schema))
        validate = lambda raw: self._chat_result(raw, schema)
        return validate(self._cached("chat", payload, self._send_chat, validate))

    @staticmethod
    def _chat_result(result, schema):
        try:
            choice = result["choices"][0]
            message = choice["message"]
            if message.get("refusal") or choice.get("finish_reason") != "stop":
                raise ResponseError("Model refused or did not finish its response")
            text = message["content"]
            if not isinstance(text, str):
                raise ResponseError("Missing completion text")
            return json.loads(text) if schema is not None else text
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ResponseError("Malformed model response") from exc

    def segment(self, image, concept):
        if not concept.strip():
            raise ValueError("Segmentation concept must not be empty")
        payload = dict(model=SAM, stream=True, input=[dict(
            type="message", role="user", content=[
                dict(type="input_text", text=concept.strip()),
                dict(type="input_image", image_url=image_data_url(image)),
            ])])
        return self._cached("segment", payload, self._send_sam)

    def _send_tool_chat(self, payload):
        return self._send_chat(payload)

    def tool_chat(self, messages, tools, *, model=SPARK):
        """Return the full assistant message, retaining tool IDs and arguments.

        Caller appends that message before its matching tool results, as required
        by the Chat Completions conversation protocol. Only auto is supported.
        """
        payload = json.loads(json.dumps(dict(model=model, messages=messages, tools=tools,
            tool_choice="auto", max_completion_tokens=MAX_COMPLETION_TOKENS), allow_nan=False))
        return self._tool_result(self._cached("tool_chat", payload, self._send_tool_chat, self._tool_result))

    @staticmethod
    def _tool_result(result):
        try:
            choice = result["choices"][0]
            message = choice["message"]
            calls = message.get("tool_calls") or []
            if message.get("refusal") or message.get("role") != "assistant":
                raise ResponseError("Model refused or returned a non-assistant message")
            if choice.get("finish_reason") == "tool_calls":
                if not calls:
                    raise ResponseError("Missing tool calls")
                ids = set()
                for call in calls:
                    if (not isinstance(call["id"], str) or not call["id"] or call["id"] in ids
                            or call.get("type") != "function"
                            or not isinstance(call["function"]["name"], str)
                            or not isinstance(call["function"]["arguments"], str)):
                        raise ResponseError("Malformed tool call")
                    ids.add(call["id"])
            elif choice.get("finish_reason") != "stop" or calls or not isinstance(message.get("content"), str):
                raise ResponseError("Model did not finish its response")
            return {key: value for key, value in message.items() if value is not None or key == "content"}
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ResponseError("Malformed tool response") from exc

    def summary(self):
        return dict(**self.guard.summary(), cache_hits=self.cache_hits, simulated=self.simulated,
                    usage=self.usage_summary())


class FakeClient(MetaClient):
    """Offline scripted client. The default abstains; it never invents PDF text.

    Fake cache keys are isolated from real API responses. Supply JSON objects
    in call order, or a handler(operation, request) for realistic test fixtures.
    """
    namespace = "fake-v1"
    simulated = True

    def __init__(self, responses=(), *, handler=None, **kwargs):
        super().__init__(**kwargs)
        self.responses = deque(responses)
        self.handler = handler
        self.requests = []

    def _fake(self, operation, payload):
        self.requests.append((operation, payload))
        if self.handler:
            return self.handler(operation, payload)
        if self.responses:
            return self.responses.popleft()
        if operation == "segment":
            return dict(boxes=[], masks=[], diagnostics=[])
        if operation == "tool_chat":
            return dict(role="assistant", content=json.dumps(dict(abstained=True, citations=[])))
        name = payload.get("response_format", {}).get("json_schema", {}).get("name")
        if name == "page_read":
            return dict(spans=[])
        if name == "records":
            return dict(records=[])
        return "FakeClient: no scripted response"

    def _send_chat(self, payload):
        result = self._fake("chat", payload)
        content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        return dict(choices=[dict(message=dict(content=content), finish_reason="stop")])

    def _send_sam(self, payload):
        return self._fake("segment", payload)

    def _send_tool_chat(self, payload):
        message = self._fake("tool_chat", payload)
        if isinstance(message, str):
            message = dict(role="assistant", content=message)
        return dict(choices=[dict(message=message,
            finish_reason="tool_calls" if message.get("tool_calls") else "stop")])
