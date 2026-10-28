import pytest

from akshara.extract import ground_records


def span(text="Amount 100", status="VERIFIED", page=1, bbox=None):
    return dict(text=text, verification=status, page=page, bbox=bbox or [10, 20, 90, 40])


def candidate(**kwargs):
    return dict(kind="amount", value="100", cells=[], evidence="Amount 100", span_ids=[0], **kwargs)


def test_citation_and_status_come_from_spans():
    data = candidate()
    data.update(page=999, bbox=[0, 0, 999, 999], verification="VERIFIED")
    records, rejected = ground_records(dict(records=[data]), [span(status="UNCERTAIN")])
    assert not rejected
    assert records[0]["page"] == 1
    assert records[0]["bbox"] == [10, 20, 90, 40]
    assert records[0]["verification"] == "UNCERTAIN"


@pytest.mark.parametrize("update", [
    dict(value="1000"), dict(evidence="Invented 100"), dict(span_ids=[8]),
    dict(span_ids=[-1]), dict(span_ids=[True]), dict(span_ids=[]), dict(span_ids=[0, 0]),
    dict(evidence=""), dict(kind="unknown"), dict(cells=["fabricated"]),
])
def test_fabrications_are_rejected(update):
    data = candidate()
    data.update(update)
    records, rejected = ground_records(dict(records=[data]), [span()])
    assert not records and len(rejected) == 1


def test_table_row_union_and_worst_status():
    spans = [span("Office 100", bbox=[10, 20, 90, 40]),
             span("Total 200", status="CORRECTED", bbox=[100, 20, 200, 40])]
    record = dict(kind="table_row", value="Office 100 Total 200", cells=["Office", "100", "Total", "200"],
                  evidence="Office 100 Total 200", span_ids=[0, 1])
    records, rejected = ground_records(dict(records=[record]), spans)
    assert not rejected
    assert records[0]["bbox"] == [10, 20, 200, 40]
    assert records[0]["verification"] == "CORRECTED"
    spans[1]["page"] = 2
    assert ground_records(dict(records=[record]), spans)[1] == ["citation crosses pages"]


def test_inferred_document_type_is_not_verified_as_a_literal():
    record = candidate()
    record.update(kind="doc_type", value="budget")
    records, _ = ground_records(dict(records=[record]), [span()])
    assert records[0]["verification"] == "UNCERTAIN"

@pytest.mark.parametrize('source,value,cells', [
    ('Amount -1000', '100', []), ('Rate 5%', '5', []),
    ('A 100 B 200', 'A 100 B 200', ['200', '100']),
    ('A 100 B 200', 'A 100 B 200', ['100', '100'])])
def test_complete_values_and_ordered_cells(source, value, cells):
    record = dict(kind='table_row' if cells else 'amount', value=value,
                  evidence=source, cells=cells, span_ids=[0])
    assert not ground_records(dict(records=[record]), [span(source)])[0]
