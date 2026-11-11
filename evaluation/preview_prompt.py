"""Show the exact prompt the model would see, in both prompt styles, without loading the model.

Examples:
  # One state, printed in full for both styles:
  python -m evaluation.preview_prompt --articles data/processed/articles_context.json \
      --start Abel_Tasman --target Pacific_Ocean

  # Prompt-length statistics over random pages:
  python -m evaluation.preview_prompt --articles data/processed/articles_context.json --stats 300
"""

import argparse
import random
from pathlib import Path
from statistics import mean, median
from types import SimpleNamespace

from transformers import AutoTokenizer

from environment.environment import WikiGameEnv
from policy.llm_policy import WikiGamePolicy

ROOT = Path(__file__).resolve().parents[1]
STYLES = ("article_text", "link_context")


def preview_policies(tokenizer, max_input_tokens):
    """Policies that only build and tokenize prompts; the model is a stand-in."""
    stand_in = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=40960),
                               device="cpu", eval=lambda: None)
    return {style: WikiGamePolicy(model=stand_in, tokenizer=tokenizer, max_input_tokens=max_input_tokens,
                                  prompt_style=style) for style in STYLES}


def prompt_ids(policy, observation):
    return policy._prepare_inputs(observation, output_tokens=64)["input_ids"][0].tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--articles", type=Path, default=ROOT / "data/processed/articles_context.json",
                        help="Articles processed by data/link_context.py.")
    parser.add_argument("--start", help="Start article ID (as in articles.json).")
    parser.add_argument("--target", help="Target article ID.")
    parser.add_argument("--stats", type=int, help="Instead, measure prompt lengths over this many random pages.")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B", help="Tokenizer to use.")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.stats is None and not (args.start and args.target):
        parser.error("Give --start and --target, or --stats N.")

    env = WikiGameEnv(args.articles)
    if "link_order" not in next(iter(env.articles.values())):
        parser.error("These articles have no link context; run data/link_context.py first.")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    policies = preview_policies(tokenizer, args.max_input_tokens)

    if args.stats is None:
        env.reset(options={"start": args.start, "target": args.target})
        observation = env.get_llm_observation()
        for style, policy in policies.items():
            ids = prompt_ids(policy, observation)
            print(f"\n{'=' * 30} {style}: {len(ids)} tokens {'=' * 30}\n")
            print(tokenizer.decode(ids))
        return

    rng = random.Random(args.seed)
    starts = [a for a, row in env.articles.items() if row["outgoing"]]
    lengths = {style: [] for style in STYLES}
    contexts_dropped = 0
    for _ in range(args.stats):
        start = rng.choice(starts)
        target = rng.choice([a for a in env.article_ids if a != start])
        env.reset(options={"start": start, "target": target})
        observation = env.get_llm_observation()
        for style, policy in policies.items():
            lengths[style].append(len(prompt_ids(policy, observation)))
        full = policies["link_context"]._encode_observation(observation)["input_ids"].shape[1]
        contexts_dropped += full > args.max_input_tokens - 64
    limit = args.max_input_tokens - 64
    for style, values in lengths.items():
        at_limit = sum(v >= limit - 8 for v in values) / len(values)
        print(f"{style:>13}: mean {mean(values):.0f} | median {median(values):.0f} | max {max(values)} "
              f"tokens | at the {args.max_input_tokens}-token limit: {at_limit:.0%}")
    print(f"link_context pages that needed sentences dropped to fit: {contexts_dropped / args.stats:.0%}")
    ratio = mean(lengths["article_text"]) / mean(lengths["link_context"])
    print(f"Average prompt is {ratio:.1f}x shorter with link_context.")


if __name__ == "__main__":
    main()
