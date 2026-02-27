#!/usr/bin/env python3
"""
audit_pairs.py — Steerability bias audit for negotiation_steering_pairs.json

Checks whether positive vs negative responses have systematic surface
differences that a steering vector could latch onto instead of learning
the intended concept.

Checks:
  1. Token/word count (positive longer? shorter?)
  2. Sentence count
  3. Opening words/phrases (first 3 words)
  4. First-person pronoun usage ("I", "I'm", "I've", "I'd", "my", "me", "we")
  5. Second-person pronoun usage ("you", "your", "you're", "you've")
  6. Hedge word count ("maybe", "perhaps", "I guess", "sort of", "I suppose",
     "possibly", "might", "could", "probably", "not sure")
  7. Apologetic language ("sorry", "I apologize", "I hate to", "I don't want to be difficult")
  8. Punctuation: question marks, exclamation marks, ellipsis
  9. Sentence-initial patterns (do positive/negative cluster differently?)
  10. Sentiment words (positive/negative valence)

Output: per-dimension summary + overall summary with effect sizes.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# ── helpers ──────────────────────────────────────────────────────────

def word_count(text: str) -> int:
    return len(text.split())

def char_count(text: str) -> int:
    return len(text)

def sentence_count(text: str) -> int:
    # split on sentence-ending punctuation
    sents = re.split(r'[.!?]+', text)
    return len([s for s in sents if s.strip()])

def count_pattern(text: str, patterns: list[str]) -> int:
    text_lower = text.lower()
    total = 0
    for p in patterns:
        total += len(re.findall(r'\b' + re.escape(p) + r'\b', text_lower))
    return total

FIRST_PERSON = ["i", "i'm", "i've", "i'd", "i'll", "my", "me", "we", "we're",
                "we've", "we'd", "our", "us"]
SECOND_PERSON = ["you", "your", "you're", "you've", "you'd", "you'll"]
HEDGE_WORDS = ["maybe", "perhaps", "i guess", "sort of", "i suppose", "possibly",
               "might", "could", "probably", "not sure", "kind of", "a bit",
               "a little", "somewhat"]
APOLOGETIC = ["sorry", "i apologize", "i hate to", "i don't want to be difficult",
              "i don't want to seem", "i don't want to cause", "please don't",
              "if that's okay", "if that would be"]
CONFIDENT = ["need", "must", "require", "expect", "non-negotiable",
             "prepared to", "committed to", "confident", "certain"]
YIELDING = ["okay", "fine", "i guess", "sure", "whatever", "i'll just",
            "i suppose", "works for me", "that's fine"]

def first_n_words(text: str, n: int = 3) -> str:
    return " ".join(text.split()[:n]).lower().rstrip(".,;:—-")

def count_questions(text: str) -> int:
    return text.count("?")

def count_exclamations(text: str) -> int:
    return text.count("!")

def count_ellipsis(text: str) -> int:
    return len(re.findall(r'\.{3}|…', text))

def count_dashes(text: str) -> int:
    return len(re.findall(r'—|--', text))


# ── main analysis ────────────────────────────────────────────────────

def analyze_pairs(pairs_file: str = None):
    if pairs_file is None:
        pairs_file = str(Path(__file__).resolve().parent.parent / "negotiation_steering_pairs.json")
    with open(pairs_file) as f:
        data = json.load(f)

    dimensions = data["dimensions"]

    # global accumulators
    all_pos_stats = []
    all_neg_stats = []

    print("=" * 80)
    print("STEERABILITY BIAS AUDIT")
    print("=" * 80)

    # track opening words globally
    pos_openers = Counter()
    neg_openers = Counter()

    for dim in dimensions:
        dim_id = dim["id"]
        dim_name = dim["name"]
        pairs = dim["pairs"]

        pos_stats = []
        neg_stats = []

        for pair in pairs:
            pos = pair["positive"]
            neg = pair["negative"]

            for text, stats_list in [(pos, pos_stats), (neg, neg_stats)]:
                stats = {
                    "words": word_count(text),
                    "chars": char_count(text),
                    "sentences": sentence_count(text),
                    "first_person": count_pattern(text, FIRST_PERSON),
                    "second_person": count_pattern(text, SECOND_PERSON),
                    "hedges": count_pattern(text, HEDGE_WORDS),
                    "apologetic": count_pattern(text, APOLOGETIC),
                    "confident": count_pattern(text, CONFIDENT),
                    "yielding": count_pattern(text, YIELDING),
                    "questions": count_questions(text),
                    "exclamations": count_exclamations(text),
                    "ellipsis": count_ellipsis(text),
                    "dashes": count_dashes(text),
                    "opener": first_n_words(text),
                }
                stats_list.append(stats)

            pos_openers[first_n_words(pos)] += 1
            neg_openers[first_n_words(neg)] += 1

        # per-dimension summary
        def avg(lst, key):
            return sum(s[key] for s in lst) / len(lst) if lst else 0

        print(f"\n{'─' * 80}")
        print(f"DIMENSION: {dim_name} ({dim_id}) — {len(pairs)} pairs")
        print(f"{'─' * 80}")

        metrics = ["words", "chars", "sentences", "first_person", "second_person",
                    "hedges", "apologetic", "confident", "yielding", "questions",
                    "exclamations", "ellipsis", "dashes"]

        print(f"{'Metric':<20} {'Positive':>10} {'Negative':>10} {'Diff':>10} {'Ratio':>8}")
        print(f"{'─' * 20} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 8}")

        for m in metrics:
            p = avg(pos_stats, m)
            n = avg(neg_stats, m)
            diff = p - n
            ratio = p / n if n > 0 else float('inf') if p > 0 else 1.0
            flag = " ⚠" if abs(diff) > 0.5 and (ratio > 1.5 or ratio < 0.67) else ""
            print(f"{m:<20} {p:>10.2f} {n:>10.2f} {diff:>+10.2f} {ratio:>7.2f}x{flag}")

        # openers for this dimension
        dim_pos_openers = Counter(s["opener"] for s in pos_stats)
        dim_neg_openers = Counter(s["opener"] for s in neg_stats)
        print(f"\n  Top positive openers: {dim_pos_openers.most_common(5)}")
        print(f"  Top negative openers: {dim_neg_openers.most_common(5)}")

        all_pos_stats.extend(pos_stats)
        all_neg_stats.extend(neg_stats)

    # ── global summary ───────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("GLOBAL SUMMARY (all 180 pairs)")
    print(f"{'=' * 80}")

    def avg(lst, key):
        return sum(s[key] for s in lst) / len(lst) if lst else 0

    metrics = ["words", "chars", "sentences", "first_person", "second_person",
               "hedges", "apologetic", "confident", "yielding", "questions",
               "exclamations", "ellipsis", "dashes"]

    print(f"\n{'Metric':<20} {'Positive':>10} {'Negative':>10} {'Diff':>10} {'Ratio':>8}")
    print(f"{'─' * 20} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 8}")

    findings = []
    for m in metrics:
        p = avg(all_pos_stats, m)
        n = avg(all_neg_stats, m)
        diff = p - n
        ratio = p / n if n > 0 else float('inf') if p > 0 else 1.0
        flag = ""
        if abs(diff) > 0.3 and (ratio > 1.3 or ratio < 0.77):
            flag = " ⚠ BIAS"
            findings.append((m, p, n, diff, ratio))
        print(f"{m:<20} {p:>10.2f} {n:>10.2f} {diff:>+10.2f} {ratio:>7.2f}x{flag}")

    # opener analysis
    print(f"\n{'─' * 80}")
    print("OPENING WORD ANALYSIS (first 3 words)")
    print(f"{'─' * 80}")
    print(f"\nTop 15 positive openers:")
    for opener, cnt in pos_openers.most_common(15):
        print(f"  {cnt:>3}x  \"{opener}\"")
    print(f"\nTop 15 negative openers:")
    for opener, cnt in neg_openers.most_common(15):
        print(f"  {cnt:>3}x  \"{opener}\"")

    # check for opener overlap
    pos_set = set(o for o, c in pos_openers.most_common(20))
    neg_set = set(o for o, c in neg_openers.most_common(20))
    shared = pos_set & neg_set
    pos_only = pos_set - neg_set
    neg_only = neg_set - pos_set

    print(f"\n  Shared openers (top 20): {len(shared)} — {shared}")
    print(f"  Positive-only openers:   {len(pos_only)} — {pos_only}")
    print(f"  Negative-only openers:   {len(neg_only)} — {neg_only}")

    # ── structural pattern check ─────────────────────────────────────
    print(f"\n{'─' * 80}")
    print("STRUCTURAL PATTERN CHECK")
    print(f"{'─' * 80}")

    # check: do positive responses start with "I" more?
    pos_starts_I = sum(1 for s in all_pos_stats if s["opener"].startswith("i ") or s["opener"] == "i")
    neg_starts_I = sum(1 for s in all_neg_stats if s["opener"].startswith("i ") or s["opener"] == "i")
    print(f"\n  Starts with 'I': positive={pos_starts_I}/{len(all_pos_stats)} "
          f"({100*pos_starts_I/len(all_pos_stats):.1f}%), "
          f"negative={neg_starts_I}/{len(all_neg_stats)} "
          f"({100*neg_starts_I/len(all_neg_stats):.1f}%)")

    # check: do negative responses start with "You're right" / "Oh" / "Okay"
    capitulation_starts = ["you're right", "oh", "okay", "fine", "sure",
                           "i guess", "well", "i don't want", "i mean"]
    for phrase in capitulation_starts:
        pos_cnt = sum(1 for s in all_pos_stats
                      if s["opener"].startswith(phrase.lower()))
        neg_cnt = sum(1 for s in all_neg_stats
                      if s["opener"].startswith(phrase.lower()))
        if pos_cnt > 0 or neg_cnt > 0:
            print(f"  Starts with '{phrase}': positive={pos_cnt}, negative={neg_cnt}")

    # check: do negatives use "..." (trailing off) more?
    pos_trailing = sum(s["ellipsis"] for s in all_pos_stats)
    neg_trailing = sum(s["ellipsis"] for s in all_neg_stats)
    print(f"\n  Ellipsis (...) total: positive={pos_trailing}, negative={neg_trailing}")

    # ── summary of findings ──────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("FLAGGED BIASES (potential steerability confounds)")
    print(f"{'=' * 80}")

    if findings:
        for m, p, n, diff, ratio in findings:
            direction = "higher in positive" if diff > 0 else "higher in negative"
            print(f"\n  ⚠ {m}: {direction}")
            print(f"    positive avg={p:.2f}, negative avg={n:.2f}, "
                  f"diff={diff:+.2f}, ratio={ratio:.2f}x")

            if m == "words":
                print(f"    → Steering vector may encode 'be verbose' rather than the intended trait")
            elif m == "hedges":
                print(f"    → Hedge words are a surface pattern; vector may encode "
                      f"'hedge less' rather than firmness/assertiveness")
            elif m == "apologetic":
                print(f"    → Apologetic language difference means vector may encode "
                      f"'don't apologize' rather than the intended trait")
            elif m == "yielding":
                print(f"    → Yielding words ('okay', 'fine', 'sure') cluster in negatives; "
                      f"vector may encode 'don't capitulate' rather than nuanced behavior")
            elif m == "confident":
                print(f"    → Confident language difference is expected but means the vector "
                      f"may encode assertive phrasing rather than deep reasoning")
            elif m == "questions":
                print(f"    → Question mark difference means some dimensions' vectors "
                      f"may encode 'ask questions' vs 'make statements'")
    else:
        print("\n  No major biases flagged at global level.")

    print(f"\n{'=' * 80}")
    print("RECOMMENDATIONS FOR P1 (Contrastive Pair Authors)")
    print(f"{'=' * 80}")
    print("""
  1. MATCH RESPONSE LENGTH: Positive and negative responses should be similar
     in word count. If positives are systematically longer, the steering vector
     will partially encode "be more verbose."

  2. VARY OPENING PHRASES: If all positive examples for a dimension start with
     "I understand..." and all negatives start with "Okay...", the vector encodes
     sentence-opening patterns, not the underlying trait.

  3. CONTROL FOR HEDGE WORDS: Some hedging in positive examples is okay.
     Zero hedging in positives + heavy hedging in negatives = surface pattern.

  4. CROSS-CONTAMINATE: Some positive examples should contain words/phrases
     that appear in negatives and vice versa, so the vector can't rely on
     simple lexical cues.

  5. CHECK EACH DIMENSION INDEPENDENTLY: Some dimensions may be fine while
     others are heavily biased. The per-dimension tables above show which.
""")


if __name__ == "__main__":
    analyze_pairs()
