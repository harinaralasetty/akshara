import asyncio
import json

import httpx
import pytest
from openai import APIStatusError

import akshara.meta_client as module
from akshara.meta_client import (CacheMiss, FakeClient, MetaClient, ResponseError,
                                 SpendGuard, SpendLimit, parse_sam_events, request_hash, sam_events)


def mock_api(monkeypatch, handler):
    from openai import OpenAI
    monkeypatch.setenv("MODEL_API_KEY", "offline-test-key")
    def factory(**kwargs):
        assert kwargs["base_url"] == "https://api.meta.ai/v1"
        assert kwargs["max_retries"] == 0
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
    client = MetaClient(tmp_path, cap="0.25")
    with pytest.raises(APIStatusError):
        client.chat("first")
    with pytest.raises(SpendLimit):
        client.chat("second")
    assert len(calls) == 1  # No hidden SDK retry.
    assert client.guard.summary()["reserved_usd"] == .25
    assert not list(tmp_path.glob("*.json"))


def test_zero_cap_can_use_cache_but_cannot_make_a_call(tmp_path, monkeypatch):
    mock_api(monkeypatch, lambda r: httpx.Response(200, json=completion()))
    MetaClient(tmp_path).chat("cached")
    client = MetaClient(tmp_path, cap=0)
    assert client.chat("cached") == "hello"
    with pytest.raises(SpendLimit):
        client.chat("new")


def test_decimal_budget_and_invalid_limits():
    guard = SpendGuard("0.3", "0.1")
    for _ in range(3):
        guard.reserve()
    with pytest.raises(SpendLimit):
        guard.reserve()
    for cap, estimate in [("NaN", 1), ("Infinity", 1), (-1, 1), (2, 0), (2, "NaN"), (2, -1)]:
        with pytest.raises(ValueError):
            SpendGuard(cap, estimate)


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
