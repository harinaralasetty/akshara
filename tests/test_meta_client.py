import asyncio
import json
from decimal import Decimal

import httpx
import pytest
from openai import APIStatusError
from tenacity import wait_none

import akshara.meta_client as module
from akshara.meta_client import (CacheMiss, FakeClient, MetaClient, ResponseError,
                                 MissingKey, SpendGuard, SpendLimit, load_api_key, parse_sam_events,
                                 request_hash, sam_events, usage_cost)


def mock_api(monkeypatch, handler):
    from openai import OpenAI
    monkeypatch.setenv("MODEL_API_KEY", "offline-test-key")
    monkeypatch.setattr(module, "wait_exponential", lambda **kwargs: wait_none())
    def factory(**kwargs):
        assert kwargs["base_url"] == "https://api.meta.ai/v1"
        assert kwargs["max_retries"] == 0
        assert kwargs["timeout"] == 60
        return OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(module, "OpenAI", factory)


def completion(content="hello", finish="stop"):
    return dict(id="test", object="chat.completion", created=0, model=module.SPARK,
                choices=[dict(index=0, message=dict(role="assistant", content=content), finish_reason=finish)])


def test_sdk_shape_cache_and_keyless_replay(tmp_path, monkeypatch):
    calls = []
    def handler(request):
        calls.append(json.loads(request.content))
        assert request.headers["authorization"] == "Bearer offline-test-key"
        return httpx.Response(200, json=completion('{"ok": true}'))
    mock_api(monkeypatch, handler)
    schema = dict(type="object", properties=dict(ok=dict(type="boolean")), required=["ok"], additionalProperties=False)
    client = MetaClient(tmp_path, cap="0.25")
    kwargs = dict(image=b"png", schema=schema, schema_name="test")
    assert client.chat("read", **kwargs) == {"ok": True}
    assert client.chat("read", **kwargs) == {"ok": True}
    assert len(calls) == 1 and client.guard.calls == 1 and client.cache_hits == 1
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert calls[0]["messages"][0]["content"][1]["image_url"]["url"] == "data:image/png;base64,cG5n"
    monkeypatch.delenv("MODEL_API_KEY")
    replay = MetaClient(tmp_path, replay=True, cap=0)
    assert replay.chat("read", **kwargs) == {"ok": True}
    assert replay.guard.calls == 0
    with pytest.raises(CacheMiss):
        replay.chat("different", **kwargs)
    assert len(calls) == 1


def test_budget_blocks_before_network_and_keeps_failed_reservations(tmp_path, monkeypatch):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(503, json=dict(error=dict(message="offline transient failure")))
    mock_api(monkeypatch, handler)
    client = MetaClient(tmp_path, cap="0.04", max_attempts=1)
    with pytest.raises(APIStatusError):
        client.chat("first")
    with pytest.raises(SpendLimit):
        client.chat("second")
    assert len(calls) == 1  # No hidden SDK retry.
    assert client.guard.summary()["reserved_usd"] == float(client.estimate(dict(
        model=module.SPARK, messages=[dict(role="user", content=[dict(type="text", text="first")])],
        max_completion_tokens=8192)))
    assert not list(tmp_path.glob("*.json"))


def test_zero_cap_can_use_cache_but_cannot_make_a_call(tmp_path, monkeypatch):
    mock_api(monkeypatch, lambda r: httpx.Response(200, json=completion()))
    MetaClient(tmp_path).chat("cached")
    client = MetaClient(tmp_path, cap=0)
    assert client.chat("cached") == "hello"
    with pytest.raises(SpendLimit):
        client.chat("new")


def test_decimal_budget_and_invalid_limits():
    guard = SpendGuard("0.3")
    for _ in range(3):
        guard.reserve("0.1")
    with pytest.raises(SpendLimit):
        guard.reserve("0.1")
    for cap in ("NaN", "Infinity", -1):
        with pytest.raises(ValueError):
            SpendGuard(cap)
    for estimate in (0, "NaN", -1):
        with pytest.raises(ValueError):
            SpendGuard(2).reserve(estimate)


def test_hashes_cover_model_images_schema_and_namespace():
    payload = dict(a=1, b=2)
    assert request_hash("real", "chat", payload) == request_hash("real", "chat", dict(b=2, a=1))
    original = request_hash("real", "chat", payload)
    for namespace, operation, request in [("fake", "chat", payload), ("real", "sam", payload),
                                          ("real", "chat", dict(a=1, b=3))]:
        assert original != request_hash(namespace, operation, request)


def test_fake_cache_cannot_satisfy_real_replay(tmp_path):
    fake = FakeClient(cache_dir=tmp_path)
    fake.chat("test")
    assert fake.guard.calls == 0
    with pytest.raises(CacheMiss):
        MetaClient(tmp_path, replay=True).chat("test")


@pytest.mark.parametrize("finish,content", [("length", "{}"), ("stop", "not json")])
def test_bad_chat_output_fails(tmp_path, monkeypatch, finish, content):
    mock_api(monkeypatch, lambda r: httpx.Response(200, json=completion(content, finish)))
    with pytest.raises(ResponseError):
        MetaClient(tmp_path).chat("test", schema={})


def test_corrupt_cache_fails_closed(tmp_path):
    fake = FakeClient(cache_dir=tmp_path)
    fake.chat("test")
    path = next(tmp_path.glob("*.json"))
    data = json.loads(path.read_text())
    data["namespace"] = "real"
    path.write_text(json.dumps(data))
    with pytest.raises(ResponseError):
        FakeClient(cache_dir=tmp_path, replay=True).chat("test")


def test_replay_miss_never_reads_key_or_sends(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("replay attempted an API request")
    monkeypatch.setattr(MetaClient, "_key", forbidden)
    client = MetaClient(tmp_path, replay=True)
    with pytest.raises(CacheMiss):
        client.chat("missing")
    with pytest.raises(CacheMiss):
        client.segment(b"png", "table")


def test_sam_request_and_replay(tmp_path):
    fake = FakeClient(cache_dir=tmp_path)
    assert fake.segment(b"png", " table ")["boxes"] == []
    operation, payload = fake.requests[0]
    assert operation == "segment"
    assert payload == dict(model="sam-3.1", stream=True, input=[dict(
        type="message", role="user", content=[dict(type="input_text", text="table"),
            dict(type="input_image", image_url="data:image/png;base64,cG5n")])])
    assert FakeClient(cache_dir=tmp_path, replay=True).segment(b"png", "table")["boxes"] == []
    with pytest.raises(ValueError):
        fake.segment(b"png", " ")


def test_sam_stream_adapter_deduplicates_lane_completion():
    async def source():
        yield dict(type="response.content_part.done", item_id="x", output_index=0,
                   content_index=0, part=dict(type="output_text", text=""))
        yield dict(type="response.output_text.done", item_id="x", output_index=0, content_index=0, text="")
    async def collect():
        return [event async for event in sam_events(source())]
    events = asyncio.run(collect())
    assert [e["type"] for e in events].count("response.output_text.done") == 1


def test_sam_parser_completed_empty_and_incomplete():
    async def source(complete):
        yield dict(type="response.created", response=dict(id="r"))
        if complete:
            yield dict(type="response.output_text.done", item_id="x", output_index=0,
                       content_index=0, text="")
            yield dict(type="response.completed", response=dict(id="r", status="completed", output=[]))
    assert asyncio.run(parse_sam_events(source(True)))["boxes"] == []
    with pytest.raises(ResponseError):
        asyncio.run(parse_sam_events(source(False)))


def test_sam_sdk_stream_decodes_boxes_masks_and_caches(tmp_path, monkeypatch):
    from openai import AsyncOpenAI
    calls = []
    # A valid SAM wire record: a 2x2 foreground mask at pixel bounds [1,2,3,4].
    text = "<0f>0<|box;x1=1;y1=2;x2=2;y2=3;w=10;h=10|><|mask;x=0;y=0;data=2,2,!!!!!&t.SKi|>\n"
    events = [dict(type="response.output_text.delta", item_id="x", output_index=0,
                   content_index=0, delta=text),
              dict(type="response.content_part.done", item_id="x", output_index=0,
                   content_index=0, part=dict(type="output_text", text=text, annotations=[])),
              dict(type="response.completed", response=dict(id="r", status="completed", output=[]))]
    def handler(request):
        calls.append(json.loads(request.content))
        assert request.url.path == "/v1/responses"
        content = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
        return httpx.Response(200, content=content, headers={"content-type": "text/event-stream"})
    def factory(**kwargs):
        assert kwargs["max_retries"] == 0
        return AsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    monkeypatch.setenv("MODEL_API_KEY", "offline-test-key")
    monkeypatch.setattr(module, "AsyncOpenAI", factory)
    client = MetaClient(tmp_path)
    result = client.segment(b"png", "text block")
    assert len(result["boxes"]) == len(result["masks"]) == 1
    assert result["boxes"][0]["left"] == 1 and result["boxes"][0]["right"] == 3
    assert result["masks"][0]["mask"]["width"] == 2
    monkeypatch.delenv("MODEL_API_KEY")
    assert MetaClient(tmp_path, replay=True).segment(b"png", "text block") == result
    assert len(calls) == 1 and client.guard.calls == 1


def test_tool_chat_preserves_assistant_calls_and_keyless_replay(tmp_path, monkeypatch):
    message = dict(role='assistant', content=None, tool_calls=[dict(id='c1',type='function',
        function=dict(name='recover_page',arguments='{"page":1}'))])
    raw=completion(None,'tool_calls'); raw['choices'][0]['message']=message
    sent=[]
    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200,json=raw)
    mock_api(monkeypatch,handler)
    messages=[dict(role='user',content='Read page')]
    tools=[dict(type='function',function=dict(name='recover_page',parameters=dict(type='object')))]
    client=MetaClient(tmp_path)
    assert client.tool_chat(messages,tools)==message
    assert sent[0]['tool_choice']=='auto' and sent[0]['tools']==tools
    monkeypatch.delenv('MODEL_API_KEY')
    assert MetaClient(tmp_path,replay=True).tool_chat(messages,tools)==message
    assert len(sent)==1
    messages.append(message)
    with pytest.raises(CacheMiss):
        MetaClient(tmp_path,replay=True).tool_chat(messages,tools)


@pytest.mark.parametrize('finish,message', [('length',dict(role='assistant',content='x')),
    ('tool_calls',dict(role='assistant',tool_calls=[])),
    ('tool_calls',dict(role='assistant',tool_calls=[dict(id='x',type='function',function=dict(name='x',arguments={}))])),
    ('stop',dict(role='assistant',content='x',refusal='no'))])
def test_tool_chat_rejects_incomplete_or_malformed_messages(tmp_path,monkeypatch,finish,message):
    raw=completion('',finish);raw['choices'][0]['message']=message
    mock_api(monkeypatch,lambda r:httpx.Response(200,json=raw))
    with pytest.raises(ResponseError):
        MetaClient(tmp_path).tool_chat([dict(role='user',content='x')],[])


@pytest.mark.parametrize("model,expected", [(module.SPARK, "0.001235"), (module.WONDER, "0.004480")])
def test_token_tariffs(model, expected):
    usage = dict(input_tokens=1000, cached_input_tokens=400, output_tokens=100)
    assert usage_cost(model, usage) == Decimal(expected)
    assert usage_cost(module.SAM, {}, images=5) == Decimal("0.0125")


def test_usage_reconciles_and_cache_does_not_bill_again(tmp_path, monkeypatch):
    raw = completion()
    raw["usage"] = dict(prompt_tokens=1000, completion_tokens=100, total_tokens=1100,
                        prompt_tokens_details=dict(cached_tokens=400))
    mock_api(monkeypatch, lambda r: httpx.Response(200, json=raw))
    client = MetaClient(tmp_path)
    assert client.guard.cap == 3
    client.chat("cost")
    assert client.guard.reserved == Decimal("0.001235")
    client.chat("cost")
    u = client.summary()["usage"]
    assert u["cost_usd"] == .001235 and u["cache_hits"] == 1 and u["paid_calls"] == 1
    assert u["input_tokens"] == 2000 and u["cached_input_tokens"] == 800
    assert u["by_model"][module.SPARK]["calls"] == 2
    replay = MetaClient(tmp_path, replay=True, cap=0)
    replay.chat("cost")
    assert replay.summary()["usage"]["cost_usd"] == 0
    assert replay.summary()["usage"]["output_tokens"] == 100


def test_unreported_usage_keeps_estimate_and_is_explicit(tmp_path, monkeypatch):
    mock_api(monkeypatch, lambda r: httpx.Response(200, json=completion()))
    client = MetaClient(tmp_path)
    client.chat("unreported")
    u = client.usage_summary()
    assert u["cost_usd"] == 0 and u["unreported_usage_calls"] == 1
    assert u["unresolved_usd"] == float(client.guard.reserved) > 0


@pytest.mark.parametrize("failure", [429, 500, 503, "timeout"])
def test_retryable_errors_back_off_and_account_each_attempt(tmp_path, monkeypatch, failure):
    requests = []
    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            if failure == "timeout":
                raise httpx.ReadTimeout("offline", request=request)
            return httpx.Response(failure, json=dict(error=dict(message="offline")))
        raw = completion()
        raw["usage"] = dict(prompt_tokens=10, completion_tokens=2, total_tokens=12)
        return httpx.Response(200, json=raw)
    mock_api(monkeypatch, handler)
    client = MetaClient(tmp_path)
    assert client.chat("retry") == "hello"
    u = client.usage_summary()
    assert len(requests) == client.guard.calls == 2 and u["failed_calls"] == 1
    assert u["unresolved_usd"] > u["cost_usd"] > 0
    assert float(client.guard.reserved) == pytest.approx(u["cost_usd"] + u["unresolved_usd"])


def test_retries_obey_cap_before_another_request(tmp_path, monkeypatch):
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(503, json=dict(error=dict(message="offline")))
    mock_api(monkeypatch, handler)
    client = MetaClient(tmp_path, cap="0.04")
    with pytest.raises(SpendLimit):
        client.chat("retry")
    assert len(requests) == 1 and client.usage_summary()["failed_calls"] == 1


def test_client_errors_are_not_retried(tmp_path, monkeypatch):
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(400, json=dict(error=dict(message="invalid")))
    mock_api(monkeypatch, handler)
    with pytest.raises(APIStatusError):
        MetaClient(tmp_path).chat("bad")
    assert len(requests) == 1


def test_reservations_follow_prompt_and_image_size(tmp_path):
    import io
    from PIL import Image
    def payload(text, size=None):
        content = [dict(type="text", text=text)]
        if size:
            png = io.BytesIO()
            Image.new("RGB", size).save(png, format="PNG")
            content.append(dict(type="image_url", image_url=dict(url=module.image_data_url(png.getvalue()))))
        return dict(model=module.SPARK, messages=[dict(role="user", content=content)], max_completion_tokens=64)
    client = MetaClient(tmp_path)
    assert client.estimate(payload("x")) < client.estimate(payload("x" * 1000))
    assert client.estimate(payload("x", (100, 100))) < client.estimate(payload("x", (1000, 1000)))
    assert client.estimate(payload("x", (100, 100))) < Decimal("0.01")
    assert client.estimate(dict(model=module.SAM, input=[dict(type="input_image", image_url="data:,a")])) == Decimal("0.0025")


def test_key_only_comes_from_environment(tmp_path, monkeypatch):
    env = tmp_path / "env"
    env.write_text('# no shell evaluation\nOTHER=ignored\nMODEL_API_KEY="offline-file-key"\n')
    monkeypatch.setenv("AKSHARA_ENV_FILE", str(env))
    with pytest.raises(MissingKey, match="MODEL_API_KEY missing"):
        load_api_key()
    monkeypatch.setenv("MODEL_API_KEY", "offline-env-key")
    assert load_api_key() == "offline-env-key"
    monkeypatch.delenv("MODEL_API_KEY")
    env.write_text("OTHER=not-a-key\n")
    with pytest.raises(MissingKey, match="MODEL_API_KEY missing"):
        load_api_key()


def test_default_env_file_is_not_read(tmp_path, monkeypatch):
    monkeypatch.delenv("AKSHARA_ENV_FILE", raising=False)
    directory = tmp_path / ".config/akshara"
    directory.mkdir(parents=True)
    (directory / "env").write_text("MODEL_API_KEY=offline-default-key\n")
    expanduser = module.Path.expanduser
    monkeypatch.setattr(module.Path, "expanduser", lambda p: directory / "env"
                        if str(p) == "~/.config/akshara/env" else expanduser(p))
    with pytest.raises(MissingKey, match="MODEL_API_KEY missing"):
        load_api_key()
