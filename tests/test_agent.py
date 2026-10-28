import base64
import io
import json

from PIL import Image
import pytest

from akshara import agent
from akshara.meta_client import FakeClient


def call(name, args, ident='c1'):
    return dict(id=ident, type='function', function=dict(name=name, arguments=json.dumps(args)))


def assistant(*calls, content=None):
    return dict(role='assistant', content=content, tool_calls=list(calls))


def final(citations, **kwargs):
    return assistant(content=json.dumps(dict(citations=citations, abstained=False, **kwargs)))


@pytest.fixture
def report(tmp_path):
    Image.new('RGB', (200, 100), 'white').save(tmp_path / 'page-001.png')
    span = dict(page=1, bbox=[10, 10, 80, 30], text='Budget amount 100', verification='VERIFIED')
    record = dict(span, value='100', evidence=span['text'], kind='amount')
    data = dict(simulated=False, pages=[dict(page=1, bounds=[0, 0, 100, 50], width=100,
                height=50, rotation=0, image='page-001.png', spans=[span])], records=[record])
    path = tmp_path / 'report.json'
    path.write_text(json.dumps(data))
    return path


def test_search_grounded_answer_and_cookbook_message_order(report, tmp_path):
    first = assistant(call('search_records', dict(query='100')))
    client = FakeClient(cache_dir=tmp_path / 'cache', responses=[first,
        final([dict(evidence_id='e1', quote='amount 100')], answer='Unsupported 999')])
    result = agent.ask(report, 'What amount?', client)
    assert result['answer'] == 'amount 100'
    assert result['steps'] == 2 and not result['abstained'] and result['simulated']
    assert result['citations'][0] == dict(evidence_id='e1', page=1, bbox=[10, 10, 80, 30],
                                        quote='amount 100', verification='UNCERTAIN', source='record')
    messages = client.requests[1][1]['messages']
    assert messages[-2] == first
    assert messages[-1]['role'] == 'tool' and messages[-1]['tool_call_id'] == 'c1'
    assert json.loads(messages[-1]['content'])['evidence'][0]['quote'] == 'Budget amount 100'


@pytest.mark.parametrize('content', ['free answer 100', '{}', '{"citations":[],"abstained":false}',
    '{"citations":[{"evidence_id":"e1","quote":"100"}],"abstained":false}'])
def test_answer_without_tool_evidence_abstains(report, tmp_path, content):
    result = agent.ask(report, 'Amount?', FakeClient(cache_dir=tmp_path/'cache', responses=[assistant(content=content)]))
    assert result['abstained'] and not result['citations']


@pytest.mark.parametrize('citation', [dict(evidence_id='e1', quote='999'),
    dict(evidence_id='missing', quote='100'), dict(evidence_id='e1', quote='100', page=2),
    dict(page=1, bbox=[0, 0, 1, 1], quote='100'), None])
def test_malformed_or_fabricated_citation_abstains(report, tmp_path, citation):
    client = FakeClient(cache_dir=tmp_path/'cache', responses=[assistant(call('recover_page',dict(page=1))), final([citation])])
    result = agent.ask(report, 'Amount?', client)
    assert result['abstained'] and not result['citations']


def test_read_region_sends_real_crop_and_grounds_read(report, tmp_path):
    client = FakeClient(cache_dir=tmp_path/'cache', responses=[assistant(call('read_region',dict(page=1,bbox=[10,10,80,30]))),
        dict(spans=[dict(text='Read 100', bbox=[0,0,1000,1000])]), final([dict(evidence_id='e1',quote='Read 100')])])
    result = agent.ask(report, 'Read?', client)
    assert result['answer'] == 'Read 100'
    assert result['citations'][0]['bbox'] == [10,10,80,30]
    assert result['citations'][0]['verification'] == 'UNCERTAIN'
    data = client.requests[1][1]['messages'][0]['content'][1]['image_url']['url'].split(',')[1]
    assert Image.open(io.BytesIO(base64.b64decode(data))).size == (140,40)


def test_rotated_region_crop_uses_saved_png_geometry(report, tmp_path):
    data=json.loads(report.read_text())
    data['pages'][0].update(rotation=90,width=50,height=100)
    report.write_text(json.dumps(data))
    Image.new('RGB',(100,200),'white').save(tmp_path/'page-001.png')
    client=FakeClient(cache_dir=tmp_path/'cache', responses=[assistant(call('read_region',dict(page=1,bbox=[10,10,80,30]))),
        dict(spans=[dict(text='100',bbox=[0,0,1000,1000])]),final([dict(evidence_id='e1',quote='100')])])
    result=agent.ask(report,'Amount?',client)
    data=client.requests[1][1]['messages'][0]['content'][1]['image_url']['url'].split(',')[1]
    assert Image.open(io.BytesIO(base64.b64decode(data))).size==(40,140)
    assert result['citations'][0]['bbox']==[10,10,80,30]


def test_tool_errors_are_returned_and_multiple_results_follow_assistant(report,tmp_path):
    first=assistant(call('read_region',dict(page=1,bbox=[0,0,999,999])),
                    call('recover_page',dict(page=1),'c2'))
    client=FakeClient(cache_dir=tmp_path/'cache',responses=[first,final([dict(evidence_id='e1',quote='100')])])
    result=agent.ask(report,'Amount?',client)
    messages=client.requests[1][1]['messages']
    assert messages[-3]==first
    assert json.loads(messages[-2]['content'])['error']
    assert messages[-2]['tool_call_id']=='c1' and messages[-1]['tool_call_id']=='c2'
    assert result['answer']=='100'


def test_segment_invokes_sam_but_geometry_cannot_support_answer(report,tmp_path):
    client=FakeClient(cache_dir=tmp_path/'cache',responses=[assistant(call('segment',dict(page=1,concept='table'))),
        dict(boxes=[dict(left=20,top=20,right=160,bottom=60,object_id='1')],masks=[],diagnostics=[]),
        final([dict(page=1,bbox=[10,10,80,30],quote='100')])])
    result=agent.ask(report,'Amount?',client)
    assert [r[0] for r in client.requests]==['tool_chat','segment','tool_chat']
    messages=client.requests[-1][1]['messages']
    assert json.loads(messages[-1]['content'])['regions'][0]['bbox']==[10,10,80,30]
    assert result['abstained']


def test_step_limit_bounds_model_and_tool_calls(report,tmp_path):
    client=FakeClient(cache_dir=tmp_path/'cache',responses=[assistant(call('recover_page',dict(page=1)))])
    result=agent.ask(report,'Amount?',client,max_steps=1)
    assert result['abstained'] and result['steps']==1 and len(client.requests)==1
    with pytest.raises(ValueError):
        agent.ask(report,'Amount?',client,max_steps=0)


def test_image_path_escape_becomes_tool_error(report,tmp_path):
    data=json.loads(report.read_text()); data['pages'][0]['image']='../outside.png'; report.write_text(json.dumps(data))
    client=FakeClient(cache_dir=tmp_path/'cache',responses=[assistant(call('read_region',dict(page=1,bbox=[10,10,80,30]))),assistant(content='{}')])
    assert agent.ask(report,'Read?',client)['abstained']
    assert json.loads(client.requests[-1][1]['messages'][-1]['content'])['error']


def test_simulated_report_cannot_produce_verified_citations(report,tmp_path):
    data=json.loads(report.read_text());data['simulated']=True;report.write_text(json.dumps(data))
    client=FakeClient(cache_dir=tmp_path/'cache',responses=[assistant(call('recover_page',dict(page=1))),
        final([dict(page=1,bbox=[10,10,80,30],quote='100')])])
    client.simulated=False
    result=agent.ask(report,'Amount?',client)
    assert not result['simulated'] and result['evidence_simulated']
    assert result['citations'][0]['verification']=='UNCERTAIN'


def test_real_report_preserves_verification_and_abstain_choice(report,tmp_path):
    client=FakeClient(cache_dir=tmp_path/'cache',responses=[assistant(call('recover_page',dict(page=1))),
        final([dict(evidence_id='e1',quote='100')])])
    client.simulated=False
    assert agent.ask(report,'Amount?',client)['citations'][0]['verification']=='VERIFIED'
    client=FakeClient(cache_dir=tmp_path/'other',responses=[assistant(call('recover_page',dict(page=1))),
        assistant(content=json.dumps(dict(abstained=True,citations=[dict(evidence_id='e1',quote='100')])))])
    assert agent.ask(report,'Amount?',client)['abstained']


def test_excessive_tool_batch_is_bounded_without_executing(report,tmp_path):
    client=FakeClient(cache_dir=tmp_path/'cache',responses=[assistant(*[
        call('read_region',dict(page=1,bbox=[10,10,80,30]),f'c{i}') for i in range(5)])])
    assert agent.ask(report,'Read?',client)['abstained']
    assert len(client.requests)==1


def test_partly_failed_read_does_not_add_unreturned_evidence(report,tmp_path):
    client=FakeClient(cache_dir=tmp_path/'cache',responses=[assistant(call('read_region',dict(page=1,bbox=[10,10,80,30]))),
        dict(spans=[dict(text='100',bbox=[0,0,1000,1000]),dict(text='bad',bbox=[-1,0,2,2])]),
        final([dict(evidence_id='e1',quote='100')])])
    result=agent.ask(report,'Amount?',client)
    assert result['abstained']
    assert json.loads(client.requests[-1][1]['messages'][-1]['content'])['error']


@pytest.mark.parametrize('rotation,expected_size',[(0,(140,40)),(90,(40,140)),(180,(140,40)),(270,(40,140))])
def test_crop_pixel_content_matches_rotation(report,tmp_path,rotation,expected_size):
    data=json.loads(report.read_text());data['pages'][0]['rotation']=rotation
    report.write_text(json.dumps(data))
    original=Image.new('RGB',(200,100),'white')
    for x in range(20,160):
        for y in range(20,60):
            original.putpixel((x,y),(255,0,0))
    original.rotate(-rotation,expand=True).save(tmp_path/'page-001.png')
    client=FakeClient(cache_dir=tmp_path/'cache',responses=[assistant(call('read_region',dict(page=1,bbox=[10,10,80,30]))),
        dict(spans=[dict(text='100',bbox=[0,0,1000,1000])]),final([dict(evidence_id='e1',quote='100')])])
    result=agent.ask(report,'Read?',client)
    encoded=client.requests[1][1]['messages'][0]['content'][1]['image_url']['url'].split(',')[1]
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as crop:
        assert crop.size==expected_size and crop.getextrema()==((255,255),(0,0),(0,0))
    assert result['citations'][0]['bbox']==[10,10,80,30]


def test_grounded_citations_without_optional_abstained_flag(report,tmp_path):
    client=FakeClient(cache_dir=tmp_path/'cache',responses=[assistant(call('recover_page',dict(page=1))),
        assistant(content=json.dumps(dict(answer='100',citations=[dict(page=1,bbox=[10,10,80,30],quote='100')])) )])
    result=agent.ask(report,'Amount?',client)
    assert result['answer']=='100' and not result['abstained']


@pytest.mark.parametrize('source,quote', [
    ('Budget amount 1000', 'amount 100'),
    ('Budget amount -100', '100'),
    ('Budget amount 1,000', '000'),
    ('Budget amount 100.5', '100'),
])
def test_cited_quote_cannot_change_part_of_a_numeric_token(report, tmp_path, source, quote):
    data = json.loads(report.read_text())
    data['pages'][0]['spans'][0]['text'] = source
    report.write_text(json.dumps(data))
    client = FakeClient(cache_dir=tmp_path / 'cache', responses=[
        assistant(call('recover_page', dict(page=1))),
        final([dict(evidence_id='e1', quote=quote)])])
    result = agent.ask(report, 'Amount?', client)
    assert result['abstained'] and not result['citations']


@pytest.mark.parametrize('rotation', [0, 90, 180, 270])
def test_full_edge_region_on_fractional_page_is_grounded(report, tmp_path, rotation):
    bounds = [0, 0, 595.2755737304688, 841.8897705078125]
    data = json.loads(report.read_text())
    data['pages'][0].update(bounds=bounds, rotation=rotation)
    report.write_text(json.dumps(data))
    size = (1684, 1191) if rotation in (90, 270) else (1191, 1684)
    Image.new('RGB', size, 'white').save(tmp_path / 'page-001.png')
    client = FakeClient(cache_dir=tmp_path / 'cache', responses=[
        assistant(call('read_region', dict(page=1, bbox=bounds))),
        dict(spans=[dict(text='100', bbox=[0, 0, 1000, 1000])]),
        final([dict(evidence_id='e1', quote='100')])])
    result = agent.ask(report, 'Read?', client)
    assert not result['abstained']
    box = result['citations'][0]['bbox']
    assert 0 <= box[0] < box[2] <= bounds[2]
    assert 0 <= box[1] < box[3] <= bounds[3]


@pytest.mark.parametrize('tool', ['read_region', 'segment'])
@pytest.mark.parametrize('failure', ['CacheMiss', 'SpendLimit'])
def test_model_operational_errors_propagate_from_tools(report, tmp_path, tool, failure):
    from akshara import meta_client

    exception = getattr(meta_client, failure)
    arguments = dict(page=1, bbox=[10, 10, 80, 30]) if tool == 'read_region' else dict(page=1, concept='table')
    def handler(operation, payload):
        if operation == 'tool_chat':
            return assistant(call(tool, arguments))
        raise exception('unavailable')
    client = FakeClient(cache_dir=tmp_path / 'cache', handler=handler)
    with pytest.raises(exception):
        agent.ask(report, 'Read?', client)

@pytest.mark.parametrize('tool', [None, 'read_region', 'segment'])
@pytest.mark.parametrize('failure', ['auth', 'connection', 'missing'])
def test_provider_failures_propagate(report, tmp_path, tool, failure):
    import httpx
    from openai import AuthenticationError, APIConnectionError
    from akshara.meta_client import MissingKey
    request = httpx.Request('POST', 'https://offline.invalid')
    error = {'auth': AuthenticationError('bad key', response=httpx.Response(401, request=request), body=None),
             'connection': APIConnectionError(request=request), 'missing': MissingKey('no key')}[failure]
    def handler(operation, payload):
        if tool and operation == 'tool_chat':
            args = dict(page=1, bbox=[10, 10, 80, 30]) if tool == 'read_region' else dict(page=1, concept='table')
            return assistant(call(tool, args))
        raise error
    client = FakeClient(cache_dir=tmp_path / 'cache', handler=handler)
    client.simulated = False
    with pytest.raises(type(error)):
        agent.ask(report, 'Read?', client)
