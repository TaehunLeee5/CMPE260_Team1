"""How much article text the current prompts cut off, and which links the model can still read about.

Every state an evaluation visited (each page shown in a prompt, weighted by how often
it was visited) is rebuilt in the default article_text prompt style. For each one this
measures how much of the article text fits, and for each link whether its title appears
in the text that fits, only in the part that was cut off, or nowhere in the plain text
(the link's wording differs from its title, or it sits in a table or list the plain
text lost). With --distances it also checks the links that lead one click closer to
the target.

Titles are matched case-insensitively, without a trailing "(...)" qualifier. Each page
is measured with a one-page route, which leaves slightly more room for text than later
steps of an episode, so truncation is, if anything, slightly understated.

Usage:
  python -m evaluation.truncation_stats --results evaluation/results/val500_base.json --distances
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median
from types import SimpleNamespace

from environment.environment import WikiGameEnv
from policy.llm_policy import WikiGamePolicy

ROOT = Path(__file__).resolve().parents[1]
TEXT_MARKER = "Current article text:\n"


def title_pattern(title):
    """Case-insensitive whole-phrase match of a title without its "(...)" qualifier."""
    core = re.sub(r"\s*\([^)]*\)\s*$", "", title).strip() or title
    return re.compile(r"(?<!\w)" + re.escape(core) + r"(?!\w)", re.IGNORECASE)


def kept_text(policy, tokenizer, observation):
    """The article text that actually fits in the prompt."""
    prompt = tokenizer.decode(policy._prepare_inputs(observation, output_tokens=64)["input_ids"][0])
    return prompt.split(TEXT_MARKER, 1)[1].split("<|im_end|>", 1)[0]


def visits_from_results(results):
    """(page, target) for every prompt shown: each page on a route except the last."""
    visits = Counter()
    for episode in results["episodes"]:
        path = episode["path"]
        for page in path[:-1]:
            visits[(page, episode["target"])] += 1
    return visits


def measure(env, policy, tokenizer, visits, distance=None):
    """distance(page, target) enables the closer-link checks; None skips them.

    Dead-end pages (no outgoing links, so only go_back) have no links to measure and
    are counted separately.
    """
    by_page, totals = {}, Counter()
    kept_fractions, truncated, dead_end = [], 0, 0
    for (page, target), count in visits.items():
        if not env.articles[page]["outgoing"]:
            dead_end += count
            continue
        if page not in by_page:
            env.reset(options={"start": page, "target": target})
            observation = env.get_llm_observation()
            full = observation["text"]
            kept = kept_text(policy, tokenizer, observation)
            full_tokens = len(tokenizer.encode(full, add_special_tokens=False))
            kept_tokens = len(tokenizer.encode(kept, add_special_tokens=False))
            status = {}
            for link in env.articles[page]["outgoing"]:
                pattern = title_pattern(env.articles[link]["title"])
                status[link] = ("kept" if pattern.search(kept)
                                else "cut" if pattern.search(full) else "not_in_text")
            by_page[page] = (kept_tokens / full_tokens if full_tokens else 1.0, kept_tokens < full_tokens - 2, status)
        fraction, was_cut, status = by_page[page]
        kept_fractions += [fraction] * count
        truncated += was_cut * count
        for value in status.values():
            totals["links_" + value] += count
        if distance is not None:
            here = distance(page, target)
            good = [link for link in status if here is not None and distance(link, target) == here - 1]
            if good:
                totals["states_with_closer_link"] += count
                seen = {status[link] for link in good}
                if "kept" in seen:
                    totals["closer_link_readable"] += count
                elif "cut" in seen:
                    totals["closer_link_only_in_cut_text"] += count
                else:
                    totals["closer_link_not_in_text"] += count
    states = sum(visits.values()) - dead_end
    links = totals["links_kept"] + totals["links_cut"] + totals["links_not_in_text"]
    report = {
        "prompts_measured": states, "prompts_on_dead_end_pages_skipped": dead_end,
        "distinct_pages": len(by_page),
        "prompts_with_text_cut": truncated / states,
        "mean_share_of_text_kept": mean(kept_fractions),
        "median_share_of_text_kept": median(kept_fractions),
        "links_named_in_kept_text": totals["links_kept"] / links,
        "links_named_only_in_cut_text": totals["links_cut"] / links,
        "links_not_named_in_plain_text": totals["links_not_in_text"] / links,
    }
    if distance is not None and totals["states_with_closer_link"]:
        closer = totals["states_with_closer_link"]
        report.update({
            "states_with_a_closer_link": closer,
            "closer_link_named_in_kept_text": totals["closer_link_readable"] / closer,
            "closer_links_named_only_in_cut_text": totals["closer_link_only_in_cut_text"] / closer,
            "closer_links_not_named_in_plain_text": totals["closer_link_not_in_text"] / closer,
        })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True,
                        help="An evaluation result file whose visited pages define the prompts measured.")
    parser.add_argument("--articles", type=Path, default=ROOT / "data/processed/articles.json")
    parser.add_argument("--distances", action="store_true",
                        help="Also check links one click closer to the target (needs the distance matrix).")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B", help="Tokenizer to use.")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    stand_in = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=40960),
                               device="cpu", eval=lambda: None)
    policy = WikiGamePolicy(model=stand_in, tokenizer=tokenizer, max_input_tokens=args.max_input_tokens)
    # Prompts are rebuilt in a plain environment, so pages that cannot reach the target
    # still load; distances come from a separate environment that has the matrix.
    env = WikiGameEnv(args.articles)
    distance = WikiGameEnv(args.articles, reward="progress").shortest_distance if args.distances else None
    results = json.loads(args.results.read_text(encoding="utf-8"))
    report = measure(env, policy, tokenizer, visits_from_results(results), distance)
    for key, value in report.items():
        print(f"{key:>40}: {value:.1%}" if isinstance(value, float) else f"{key:>40}: {value}")


if __name__ == "__main__":
    main()
