import json

from data import preprocess


def test_read_lines(tmp_path):
    """Read article names while skipping comments and blank lines."""
    path = tmp_path / "articles.tsv"
    path.write_text("# Articles\n\nAlbert_Einstein\nPhysics\n", encoding="utf-8")

    assert preprocess.read_lines(path) == ["Albert_Einstein", "Physics"]


def test_load_graph(tmp_path):
    """Load article IDs and their outgoing links."""
    (tmp_path / "articles.tsv").write_text("Albert_Einstein\nPhysics\n", encoding="utf-8")
    (tmp_path / "links.tsv").write_text("Albert_Einstein\tPhysics\n", encoding="utf-8")

    article_ids, outgoing = preprocess.load_graph(tmp_path)

    assert article_ids == ["Albert_Einstein", "Physics"]
    assert outgoing == {"Albert_Einstein": {"Physics"}, "Physics": set()}


def test_load_text_files(tmp_path):
    """Find article text files and decode their filenames."""
    path = tmp_path / "%C3%85land.txt"
    path.write_text("An island group.", encoding="utf-8")

    files = preprocess.load_text_files(tmp_path)

    assert files == {"Åland": path}


def test_build_articles(tmp_path):
    """Combine an article's title, text, and outgoing links."""
    path = tmp_path / "Albert_Einstein.txt"
    path.write_text("A theoretical physicist.", encoding="utf-8")

    articles = preprocess.build_articles(
        ["Albert_Einstein"],
        {"Albert_Einstein": set()},
        {"Albert_Einstein": path},
    )

    assert articles["Albert_Einstein"] == {
        "title": "Albert Einstein",
        "text": "A theoretical physicist.",
        "outgoing": [],
    }


def test_save_data(tmp_path):
    """Save articles and their order as readable JSON files."""
    articles = {"Physics": {"title": "Physics", "text": "Science.", "outgoing": []}}
    article_ids = ["Physics"]
    output = tmp_path / "processed"

    preprocess.save_data(articles, article_ids, output)

    assert json.loads((output / "articles.json").read_text(encoding="utf-8")) == articles
    assert json.loads((output / "article_order.json").read_text(encoding="utf-8")) == article_ids
