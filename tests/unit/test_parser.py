import pytest

from app.ingestion.parser import parse_feed

FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <opensearch:totalResults>2</opensearch:totalResults>
  <opensearch:startIndex>0</opensearch:startIndex>
  <opensearch:itemsPerPage>2</opensearch:itemsPerPage>
  <entry>
    <id>http://arxiv.org/abs/2601.00001v2</id>
    <updated>2026-01-05T10:00:00Z</updated>
    <published>2026-01-02T09:00:00Z</published>
    <title>A Study of Attention</title>
    <summary>  We study attention.  </summary>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
    <arxiv:doi>10.1000/xyz</arxiv:doi>
    <arxiv:journal_ref>JMLR 1 (2026)</arxiv:journal_ref>
    <arxiv:primary_category term="cs.LG"/>
    <category term="cs.LG"/>
    <category term="cs.AI"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2601.00002v1</id>
    <updated>2026-01-03T10:00:00Z</updated>
    <published>2026-01-03T10:00:00Z</published>
    <title>No DOI here</title>
    <summary>Short.</summary>
    <author><name>Grace Hopper</name></author>
    <arxiv:primary_category term="cs.CL"/>
    <category term="cs.CL"/>
  </entry>
</feed>"""


def test_parse_feed_extracts_structured_fields():
    feed = parse_feed(FEED)
    assert feed.meta.total_results == 2
    first = feed.papers[0]
    assert first.arxiv_id == "2601.00001"  # version suffix stripped
    assert first.title == "A Study of Attention"
    assert first.authors == ["Ada Lovelace", "Alan Turing"]
    assert first.primary_category == "cs.LG"
    assert first.categories == ["cs.LG", "cs.AI"]
    assert first.doi == "10.1000/xyz" and first.journal_ref == "JMLR 1 (2026)"
    assert first.abstract == "We study attention."


def test_parse_feed_missing_doi_and_journal_ref_are_none():
    second = parse_feed(FEED).papers[1]
    assert second.doi is None and second.journal_ref is None


def test_parse_feed_rejects_malformed_xml():
    with pytest.raises(ValueError):
        parse_feed(b"<feed><entry>")
