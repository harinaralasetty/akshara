"""All paid calls live here. Cache hits and replay never construct an API client."""

import asyncio
import base64
import hashlib
import json
import os
from collections import deque
from contextlib import aclosing
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from tempfile import NamedTemporaryFile

from openai import AsyncOpenAI, OpenAI

BASE_URL = "https://api.meta.ai/v1"
SPARK = "muse-spark-1.3"
WONDER = "muse-wonder-1.0"
# Budget reservation, NOT a tariff or a guarantee of actual provider billing.
# Tune this one conservative estimate for the bounded requests used in this round.
CALL_ESTIMATE_USD = "0.25"
MAX_COMPLETION_TOKENS = 8192


class CacheMiss(RuntimeError):
    pass


class SpendLimit(RuntimeError):
    pass


class ResponseError(ValueError):
    pass


class SpendGuard:
    def __init__(self, cap=2, estimate=CALL_ESTIMATE_USD):
        self.cap, self.estimate = Decimal(str(cap)), Decimal(str(estimate))
        if not self.cap.is_finite() or self.cap < 0:
            raise ValueError("Spend cap must be finite and nonnegative")
        if not self.estimate.is_finite() or self.estimate <= 0:
            raise ValueError("Call estimate must be finite and positive")
        self.reserved = Decimal(0)
        self.calls = 0

    def reserve(self):
        if self.reserved + self.estimate > self.cap:
            raise SpendLimit(f"Call would exceed USD {self.cap} run cap (reserved {self.reserved})")
        # Failed requests may have been billed: never refund their reservation.
        self.reserved += self.estimate
        self.calls += 1

    def summary(self):
        return dict(cap_usd=float(self.cap), call_estimate_usd=float(self.estimate),
                    reserved_usd=float(self.reserved), attempted_calls=self.calls)


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
        raise ResponseError(f"SAM stream failed: {exc.code}") from exc
    if result.outcome.status != "completed" or any(d.severity == "error" for d in result.diagnostics):
        raise ResponseError("SAM returned an incomplete or invalid segmentation")
    return dict(boxes=[asdict(r) for r in result.records if r.kind == "box"],
                masks=[asdict(r) for r in result.records if r.kind == "mask"],
                diagnostics=[asdict(d) for d in result.diagnostics])


class MetaClient:
    namespace = "meta-v1"
    simulated = False

    def __init__(self, cache_dir="cache", *, replay=False, cap=2, estimate=CALL_ESTIMATE_USD):
        self.cache_dir = Path(cache_dir)
        self.replay = replay
        self.guard = SpendGuard(cap, estimate)
        self.cache_hits = 0

    @staticmethod
    def _key():
        key = os.environ.get("MODEL_API_KEY")
        if not key:
            raise RuntimeError("Set MODEL_API_KEY in the environment, or use --replay / --fake")
        return key

    def _cached(self, operation, payload, send):
        digest = request_hash(self.namespace, operation, payload)
        path = self.cache_dir / f"{digest}.json"
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("request_hash") != digest or cached.get("namespace") != self.namespace:
                raise ResponseError("Invalid cache envelope")
            self.cache_hits += 1
            return cached["response"]
        if self.replay:
            raise CacheMiss(f"Replay cache miss: {operation} {digest}")
        result = send(payload)
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
        self.guard.reserve()
        # Disable hidden retries so one reservation corresponds to one attempt.
        with OpenAI(base_url=BASE_URL, api_key=key, max_retries=0, timeout=120) as api:
            response = api.chat.completions.create(**payload)
        return response.model_dump(mode="json")

    async def _sam(self, payload):
        key = self._key()
        self.guard.reserve()
        async with AsyncOpenAI(base_url=BASE_URL, api_key=key, max_retries=0, timeout=120) as api:
            stream = await api.responses.create(**payload)
            async with stream, aclosing(sam_events(stream)) as events:
                return await parse_sam_events(events)

    def _send_sam(self, payload):
        return asyncio.run(self._sam(payload))

    def chat(self, prompt, *, image=None, schema=None, schema_name="result", model=SPARK):
        content = [{"type": "text", "text": prompt}]
        if image is not None:
            content.append({"type": "image_url", "image_url": {"url": image_data_url(image)}})
        payload = dict(model=model, messages=[dict(role="user", content=content)],
                       max_completion_tokens=MAX_COMPLETION_TOKENS)
        if schema is not None:
            payload["response_format"] = dict(type="json_schema", json_schema=dict(
                name=schema_name, strict=True, schema=schema))
        result = self._cached("chat", payload, self._send_chat)
        try:
            choice = result["choices"][0]
            message = choice["message"]
            if message.get("refusal") or choice.get("finish_reason") != "stop":
                raise ResponseError("Model refused or did not finish its response")
            text = message["content"]
            return json.loads(text) if schema is not None else text
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ResponseError("Malformed model response") from exc

    def segment(self, image, concept):
        if not concept.strip():
            raise ValueError("Segmentation concept must not be empty")
        payload = dict(model="sam-3.1", stream=True, input=[dict(
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
        result = self._cached("tool_chat", payload, self._send_tool_chat)
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
        return dict(**self.guard.summary(), cache_hits=self.cache_hits, simulated=self.simulated)


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
