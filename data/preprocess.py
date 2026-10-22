import argparse
import json
from pathlib import Path
from urllib.parse import unquote


def read_lines(filename):
    """Read nonempty lines from a file, skipping lines that start with #."""
    lines = []
    with filename.open(encoding="utf-8-sig") as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    return lines


def load_graph(graph_folder):
    """Load article IDs in their original order and find each article's outgoing links."""
    article_ids = read_lines(graph_folder / "articles.tsv")
    if not article_ids or len(article_ids) != len(set(article_ids)):
        raise ValueError("The article list is empty or contains duplicate IDs.")

    outgoing = {}
    for article_id in article_ids:
        outgoing[article_id] = set()

    for line in read_lines(graph_folder / "links.tsv"):
        source, target = line.split("\t")
        if source not in outgoing or target not in outgoing:
            raise ValueError(f"Unknown article in link: {source} -> {target}")
        outgoing[source].add(target)

    return article_ids, outgoing


def load_text_files(text_folder):
    """Find text files in the folder and its subfolders, using decoded names as keys."""
    if not text_folder.is_dir():
        raise ValueError(f"Text folder not found: {text_folder}")

    text_files = {}
    for path in text_folder.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".txt":
            name = unquote(path.stem)
            if name in text_files:
                raise ValueError(f"More than one text file found for: {name}")
            text_files[name] = path

    return text_files


def build_articles(article_ids, outgoing, text_files):
    """Combine titles, text, and links, raising an error for missing or empty text."""
    articles = {}
    missing = []

    for article_id in article_ids:
        name = unquote(article_id)
        if name not in text_files:
            missing.append(article_id)
            continue

        text = text_files[name].read_text(encoding="utf-8-sig")
        if not text.strip():
            raise ValueError(f"Empty article text: {article_id}")

        articles[article_id] = {
            "title": name.replace("_", " "),
            "text": text,
            "outgoing": sorted(outgoing[article_id]),
        }

    if missing:
        raise ValueError(
            f"Missing text for {len(missing)} articles. "
            f"Examples: {missing[:5]}. Check text_folder and the filenames."
        )

    return articles


def save_data(articles, article_ids, output_folder):
    """Save the combined articles and original article order as JSON files."""
    output_folder.mkdir(parents=True, exist_ok=True)

    with (output_folder / "articles.json").open("w", encoding="utf-8") as file:
        json.dump(articles, file, ensure_ascii=False, indent=2)

    with (output_folder / "article_order.json").open("w", encoding="utf-8") as file:
        json.dump(article_ids, file, ensure_ascii=False, indent=2)


def main():
    """Read folder paths from command-line arguments and run preprocessing."""
    parser = argparse.ArgumentParser(description="Prepare the Wikispeedia dataset.")
    parser.add_argument(
        "--graph-folder", type=Path, required=True,
        help="Folder containing articles.tsv and links.tsv",
    )
    parser.add_argument(
        "--text-folder", type=Path, required=True,
        help="Folder containing the article text files",
    )
    args = parser.parse_args()
    output_folder = Path(__file__).resolve().parent / "processed"

    article_ids, outgoing = load_graph(args.graph_folder)
    text_files = load_text_files(args.text_folder)
    articles = build_articles(article_ids, outgoing, text_files)
    save_data(articles, article_ids, output_folder)

    link_count = sum(len(links) for links in outgoing.values())
    print(f"Saved {len(articles)} articles and {link_count} links.")
    print(f"Output folder: {output_folder}")

if __name__ == "__main__":
    main()
