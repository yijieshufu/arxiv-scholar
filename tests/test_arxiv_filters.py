"""ArXiv 搜索筛选（日期查询等）单元测试"""
from datetime import date

from src.arxiv_client import (
    PaperMeta,
    build_date_query_clause,
    _append_query_clause,
    _paper_matches_date_range,
)


class TestDateQueryClause:
    def test_submitted_range(self):
        clause = build_date_query_clause("2024-01-15", "2024-06-30", "submitted")
        assert clause == "submittedDate:[202401150000 TO 202406302359]"

    def test_updated_open_start(self):
        clause = build_date_query_clause(date_to=date(2023, 12, 31), date_field="updated")
        assert clause == "lastUpdatedDate:[* TO 202312312359]"

    def test_append_to_query(self):
        q = _append_query_clause("transformer", build_date_query_clause("2024-01-01", "2024-12-31"))
        assert q.startswith("(transformer) AND submittedDate:")


class TestClientDateFilter:
    def _paper(self, published: str) -> PaperMeta:
        return PaperMeta(
            arxiv_id="2401.00001",
            title="Test",
            authors=["A"],
            abstract="abs",
            published=published,
            updated=published,
            categories=["cs.AI"],
            pdf_url="http://example.com/pdf",
        )

    def test_in_range(self):
        p = self._paper("2024-05-10T12:00:00+00:00")
        assert _paper_matches_date_range(p, "2024-01-01", "2024-12-31")

    def test_out_of_range(self):
        p = self._paper("2023-01-01T00:00:00+00:00")
        assert not _paper_matches_date_range(p, "2024-01-01", "2024-12-31")
