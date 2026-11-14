"""Add link context from the plain article text to a copy of articles.json.

For every article this records:
  lead          the opening paragraph (the first paragraph of running text),
  link_order    outgoing links in the order the text first mentions them,
  link_context  for each mentioned link, the first sentence that names it, capped in length.

A link is "mentioned" when its title appears in the text (any capitalization, without
a trailing "(...)" qualifier: "Mercury (planet)" matches "mercury"). Its sentence is
chosen in order of preference: running prose with the title's own capitalization, then
prose in any capitalization, then anything else (such as an infobox row). Image
captions ("caption Enlarge caption") are removed, hatnotes are skipped, and navigation
lines such as "Centuries: 17th century - 18th century - 19th century" never serve as a
link's sentence. The lead is the first paragraph of prose. Prose is told apart from
infobox rows by its common function words ("the", "of", "by", "was", ...): a row such
as "Born January 23, 1832 Paris Died ..." has almost none.

Links the text never names keep only their title in the prompt. The action set is
unchanged: only links that are already outgoing links are used.

Usage:
  python -m data.link_context --output data/processed/articles_context.json
"""

import argparse
import json
import re
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
CONTEXT_WORDS = 30
LEAD_WORDS = 60
LEAD_MIN_WORDS = 12
BOILERPLATE = ("#copyright", "Schools Wikipedia", "Related subjects", "Retrieved from",
               "This reference article is mainly selected", "SOS Children",
               "Articles with similar titles", "For other uses", "redirects here", "This article is about")
SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9\"'(])")


def _words(text):
    return re.sub(r"\s+", " ", text).strip()


def remove_captions(text):
    """Delete image captions, which the plain text stores as "caption Enlarge caption"."""
    while True:
        match = re.search(r"\s+Enlarge\s+", text)
        if not match:
            return text
        before, after = text[:match.start()], text[match.end():]
        repeat = next((k for k in range(min(len(before), len(after), 400), 4, -1)
                       if before[-k:] == after[:k]), 0)
        text = _words(before[:len(before) - repeat] + " " + after[repeat:])


FUNCTION_WORDS = frozenset("""a an the of and or in on at to by for from with as into than that which who
whom whose is are was were be been being has have had it its his her their this these those he she they
but not after before during under over between about against while when where also first""".split())
PROSE_MIN_FUNCTION_WORDS = 0.15


def is_prose(text):
    """Running text rather than an infobox row: at least 15% common function words.

    "1703: Saint Petersburg founded by Peter the Great." counts; "Born January 23,
    1832 Paris Died April 30, 1883 Paris" does not.
    """
    words = [w.lower() for w in re.findall(r"[^\W\d_][\w'’-]*", text)]
    if len(words) < 4:
        return False
    return sum(w in FUNCTION_WORDS for w in words) / len(words) >= PROSE_MIN_FUNCTION_WORDS


def is_navigation(text):
    """Navigation lines like "Centuries: 17th century - 18th century - 19th century"."""
    return text.count(" - ") >= 2


def paragraphs(text, title):
    """Paragraphs of running text: blank-line separated, boilerplate and captions removed."""
    result = []
    for block in re.split(r"\n\s*\n", text):
        for item in re.split(r"(?:^|\n)\s*\*\s+", block):  # Each "* ..." list entry stands alone.
            clean = remove_captions(_words(item))
            if (not clean or clean == title or clean == "Enlarge"
                    or any(marker in clean for marker in BOILERPLATE)):
                continue
            result.append(clean)
    return result


def title_pattern(title, ignore_case=True):
    """Whole-phrase match of a title without its "(...)" qualifier."""
    core = re.sub(r"\s*\([^)]*\)\s*$", "", title).strip() or title
    return re.compile(r"(?<!\w)" + re.escape(core) + r"(?!\w)", re.IGNORECASE if ignore_case else 0)


def _window(sentence, start):
    """The sentence, cut to CONTEXT_WORDS words around the mention at character `start`."""
    words, before = sentence.split(), len(sentence[:start].split())
    if len(words) <= CONTEXT_WORDS:
        return sentence
    first = max(0, min(before - CONTEXT_WORDS // 3, len(words) - CONTEXT_WORDS))
    last = first + CONTEXT_WORDS
    return ("..." if first else "") + " ".join(words[first:last]) + ("..." if last < len(words) else "")


def extract(text, title, links):
    """Return lead, link_order and link_context for one article.

    links maps each outgoing article ID to its title.
    """
    blocks = paragraphs(text, title)
    # The lead is the first paragraph of real prose: long enough and ending like a sentence.
    lead = next((p for p in blocks if len(p.split()) >= LEAD_MIN_WORDS and is_prose(p)
                 and not is_navigation(p) and re.search(r"[.!?][\"')\]]*$", p)), None)
    if lead is not None and len(lead.split()) > LEAD_WORDS:
        lead = " ".join(lead.split()[:LEAD_WORDS]) + "..."
    sentences = [s for block in blocks for s in SENTENCE_END.split(block) if s.strip()]
    prose = [is_prose(s) for s in sentences]
    navigation = [is_navigation(s) for s in sentences]
    first_mention, context = {}, {}
    for link, link_title in links.items():
        exact, loose = title_pattern(link_title, ignore_case=False), title_pattern(link_title)
        best = None  # (preference, sentence index, match start): lower is better.
        for index, sentence in enumerate(sentences):
            match = loose.search(sentence)
            if not match:
                continue
            first_mention.setdefault(link, (index, match.start()))
            if navigation[index]:
                continue  # Counts for page order, but is never a link's sentence.
            exact_match = exact.search(sentence)
            preference = 0 if prose[index] and exact_match else 1 if prose[index] else 2
            start = exact_match.start() if exact_match else match.start()
            if best is None or preference < best[0]:
                best = (preference, index, start)
            if preference == 0:
                break
        if best is not None:
            window = _window(sentences[best[1]], best[2])
            if len(window.split()) >= 3:
                context[link] = window
    order = sorted(first_mention, key=lambda link: first_mention[link])
    return {"lead": lead, "link_order": order, "link_context": {l: context[l] for l in order if l in context}}


def add_context(articles):
    """Return a copy of articles with context fields, plus coverage statistics."""
    result, links, ordered, with_context, with_lead, words = {}, 0, 0, 0, 0, []
    for article_id, article in articles.items():
        row = dict(article)
        titles = {link: articles[link]["title"] for link in article["outgoing"] if link in articles}
        row.update(extract(article.get("text", ""), article["title"], titles))
        links += len(article["outgoing"])
        ordered += len(row["link_order"])
        with_context += len(row["link_context"])
        with_lead += row["lead"] is not None
        words += [len(text.split()) for text in row["link_context"].values()]
        result[article_id] = row
    stats = {
        "articles": len(articles), "articles_with_lead": with_lead,
        "outgoing_links": links, "links_mentioned_in_text": ordered, "links_with_context": with_context,
        "share_links_with_context": with_context / links if links else 0.0,
        "mean_context_words": mean(words) if words else 0.0,
    }
    return result, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--articles", type=Path, default=ROOT / "data/processed/articles.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/articles_context.json")
    parser.add_argument("--examples", type=int, default=3, help="Articles to print as examples.")
    args = parser.parse_args()
    if args.output.resolve() == args.articles.resolve():
        parser.error("Write to a new file; articles.json stays unchanged.")
    if args.output.exists():
        parser.error(f"{args.output} already exists.")
    articles = json.loads(args.articles.read_text(encoding="utf-8"))
    result, stats = add_context(articles)
    args.output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    shown = [a for a in result.values() if len(a["link_context"]) >= 5][:args.examples]
    for article in shown:
        print(f"\n=== {article['title']} ===\nLead: {article['lead']}")
        for link in article["link_order"][:5]:
            title = articles.get(link, {}).get("title", link)
            print(f"  - {title}: {article['link_context'].get(link)}")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
