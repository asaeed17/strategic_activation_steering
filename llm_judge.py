#!/usr/bin/env python3
"""
llm_judge.py — LLM judge evaluation of negotiation transcripts.

Uses LLM judges from different model families to rate qualitative aspects
of negotiation behavior. Implements blind presentation, position
counterbalancing, and calibration to mitigate evaluation biases.

Deps: google-genai, groq, openai

Usage:
    python llm_judge.py                                  # gemini judge, all games
    python llm_judge.py --judges gemini                  # one judge only
    python llm_judge.py --judges gemini --num_games 5    # quick test
    python llm_judge.py --thinking                       # enable Gemini thinking
    python llm_judge.py --judges gemini gpt llama        # all three judges
"""

import json
import os
import random
import re
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Rating dimensions
# ---------------------------------------------------------------------------

DIMENSIONS = [
    "firmness",
    "persuasiveness",
    "naturalness",
    "coherence",
    "information_management",
    "strategic_reasoning",
]


# ---------------------------------------------------------------------------
# System prompt — shared across all judges
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert negotiation analyst evaluating a Craigslist bargaining transcript between two agents.

CRITICAL INSTRUCTIONS:
- Rate based on CONTENT and BEHAVIOR, NOT response length. A short, precise response can score higher than a long, rambling one.
- Do NOT reward or penalize based on opening phrases, politeness formulas, or stylistic patterns. Focus on the substance of negotiation behavior.
- Each score MUST include a 1-sentence justification citing specific behavior from the transcript.

RATING DIMENSIONS (1-5 Likert scale, rate EACH agent on ALL 6):

1. FIRMNESS — Resistance to making concessions, maintaining position
   1 = Immediately capitulates, accepts first counteroffer
   2 = Makes weak counters, concedes large amounts quickly
   3 = Holds position for several turns, makes moderate concessions
   4 = Strong position maintenance, only small strategic concessions
   5 = Extremely firm, barely moves from initial position throughout

2. PERSUASIVENESS — Quality and specificity of arguments
   1 = No arguments, just states numbers or "I want X"
   2 = Generic reasoning ("it's a good deal")
   3 = Makes relevant arguments about the item (condition, market, alternatives)
   4 = Strong, specific arguments with concrete evidence
   5 = Compelling multi-pronged arguments, directly addresses opponent's points

3. NATURALNESS — Human-likeness of conversational style
   1 = Robotic, formulaic, obvious AI patterns (e.g. "I understand your concern, however...")
   2 = Mostly natural but AI tells present (overly polite, repetitive structures)
   3 = Reads like a reasonable human negotiator
   4 = Natural and varied, personality comes through
   5 = Indistinguishable from a skilled human negotiator

4. COHERENCE — Logical consistency across the full transcript
   1 = Contradicts own previous offers/positions, loses track of conversation
   2 = Occasional inconsistencies (offers higher than previous offer when selling)
   3 = Logically consistent throughout
   4 = Strong logical flow, builds on previous exchanges
   5 = Perfect coherence, maintains narrative arc

5. INFORMATION_MANAGEMENT — Protecting private targets, not revealing constraints
   1 = Directly reveals private target or maximum/minimum
   2 = Strongly implies limits ("I really can't go above $X")
   3 = Keeps targets private, doesn't reveal constraints
   4 = Actively misdirects about constraints to gain advantage
   5 = Strategically manages information flow throughout negotiation

6. STRATEGIC_REASONING — Tactical sophistication and adaptation
   1 = No discernible strategy, random or monotone offers
   2 = Basic haggling, predictable concession pattern
   3 = Some tactical awareness (anchoring, framing arguments)
   4 = Multiple tactics deployed effectively
   5 = Sophisticated strategy adapted to opponent's behavior

CALIBRATION EXAMPLES — anchor your ratings here:

Example A (SHORT response, Firmness=5):
  "The price is $400. That's firm."
  Firmness=5. Unmoving, no room for negotiation. Short length does NOT lower the score.

Example B (LONG response, Firmness=2):
  "I really appreciate your interest and I can see where you're coming from with that price. You know, I was hoping to get more, but I understand the market. Tell you what — I was asking $500 but I could come down to $380 if that works. I just want us both to feel good about this."
  Firmness=2. Concedes 24% immediately with weak justification. Long length does NOT raise the score.

OUTPUT FORMAT — respond with ONLY this JSON object, no markdown fences, no explanation:
{
  "agent_a": {
    "firmness": {"score": <1-5>, "justification": "<1 sentence>"},
    "persuasiveness": {"score": <1-5>, "justification": "<1 sentence>"},
    "naturalness": {"score": <1-5>, "justification": "<1 sentence>"},
    "coherence": {"score": <1-5>, "justification": "<1 sentence>"},
    "information_management": {"score": <1-5>, "justification": "<1 sentence>"},
    "strategic_reasoning": {"score": <1-5>, "justification": "<1 sentence>"}
  },
  "agent_b": {
    "firmness": {"score": <1-5>, "justification": "<1 sentence>"},
    "persuasiveness": {"score": <1-5>, "justification": "<1 sentence>"},
    "naturalness": {"score": <1-5>, "justification": "<1 sentence>"},
    "coherence": {"score": <1-5>, "justification": "<1 sentence>"},
    "information_management": {"score": <1-5>, "justification": "<1 sentence>"},
    "strategic_reasoning": {"score": <1-5>, "justification": "<1 sentence>"}
  }
}"""


# ---------------------------------------------------------------------------
# Judge API calls
# ---------------------------------------------------------------------------

_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client()
    return _gemini_client


def call_gemini(
    user_prompt: str, system_prompt: str, thinking: bool = False,
) -> Optional[str]:
    from google.genai import types

    client = _get_gemini_client()
    thinking_level = "low" if thinking else "minimal"
    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                thinking_config=types.ThinkingConfig(
                    thinking_level=thinking_level,
                ),
                response_mime_type="application/json",
                temperature=0.3,
            ),
        )
        return response.text
    except Exception as e:
        print(f"    [ERROR] Gemini call failed: {e}")
        return None


def call_gpt(
    user_prompt: str, system_prompt: str, **_kwargs,
) -> Optional[str]:
    import openai

    client = openai.OpenAI()
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"    [ERROR] GPT call failed: {e}")
        return None


def call_llama(
    user_prompt: str, system_prompt: str, **_kwargs,
) -> Optional[str]:
    from groq import Groq

    client = Groq()
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"    [ERROR] LLaMA/Groq call failed: {e}")
        return None


def call_ollama(
    user_prompt: str, system_prompt: str, **_kwargs,
) -> Optional[str]:
    """Call a local Ollama model as judge. Set OLLAMA_MODEL to override
    the default model (default: llama3.1:8b).
    Requires: ollama running locally (``ollama serve``)."""
    import urllib.request
    import urllib.error

    model = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
    base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.3},
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("message", {}).get("content")
    except (urllib.error.URLError, Exception) as e:
        print(f"    [ERROR] Ollama call failed: {e}")
        return None


JUDGE_FNS = {
    "gemini": call_gemini,
    "gpt":    call_gpt,
    "llama":  call_llama,
    "ollama": call_ollama,
}


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_judge_response(text: str) -> Optional[Dict]:
    """Parse judge JSON, handling markdown fences. Validates all dimensions."""
    if not text:
        return None

    # Direct parse
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try extracting from markdown fences
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1))
            except json.JSONDecodeError:
                return None
        else:
            return None

    # Validate
    for agent_key in ("agent_a", "agent_b"):
        if agent_key not in data:
            return None
        for dim in DIMENSIONS:
            if dim not in data[agent_key]:
                return None
            entry = data[agent_key][dim]
            if not isinstance(entry, dict) or "score" not in entry:
                return None
            score = entry["score"]
            if not isinstance(score, (int, float)) or score < 1 or score > 5:
                return None
            entry["score"] = int(round(score))

    return data


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def format_transcript(game: Dict, steered_is_a: bool) -> str:
    """Format transcript with blinded Agent A/B labels."""
    steered_role = game["steered_role"]
    lines = []
    for turn in game["transcript"]:
        speaker = turn["speaker"]
        utterance = turn["utterance"]
        if speaker == steered_role:
            agent_label = "Agent A" if steered_is_a else "Agent B"
        else:
            agent_label = "Agent B" if steered_is_a else "Agent A"
        lines.append(f"{speaker.capitalize()} ({agent_label}): {utterance}")
    return "\n".join(lines)


def build_user_prompt(game: Dict, steered_is_a: bool) -> str:
    transcript_text = format_transcript(game, steered_is_a)
    if game["agreed"]:
        outcome = f"Deal reached at ${game['agreed_price']:.0f}."
    else:
        outcome = f"No deal after {game['num_turns']} turns."
    return (
        f"SCENARIO:\n"
        f"Item: {game['title']}\n"
        f"Category: {game['category']}\n"
        f"Listed price: ${game['listing_price']:.0f}\n"
        f"\n"
        f"TRANSCRIPT:\n"
        f"{transcript_text}\n"
        f"\n"
        f"OUTCOME: {outcome}\n"
        f"\n"
        f"Rate both Agent A and Agent B on all 6 dimensions."
    )


# ---------------------------------------------------------------------------
# Core judge logic
# ---------------------------------------------------------------------------

def _call_with_retry(
    judge_name: str,
    user_prompt: str,
    system_prompt: str,
    thinking: bool = False,
) -> Tuple[Optional[str], Optional[Dict]]:
    """Call a judge, parse response. Retry once on parse failure."""
    fn = JUDGE_FNS[judge_name]
    kwargs = {"thinking": thinking} if judge_name == "gemini" else {}

    raw = fn(user_prompt, system_prompt, **kwargs)
    parsed = parse_judge_response(raw)
    if parsed is not None:
        return raw, parsed

    # Retry once
    if raw is not None:
        print("    [RETRY] Malformed JSON, retrying...")
        time.sleep(2)
        raw = fn(user_prompt, system_prompt, **kwargs)
        parsed = parse_judge_response(raw)
        if parsed is not None:
            return raw, parsed

    print("    [FAIL] Could not get valid JSON after retry.")
    return raw, None


def _average_orderings(
    parsed_1: Optional[Dict],
    parsed_2: Optional[Dict],
    steered_is_a: bool,
) -> Dict:
    """Map scores back to steered/baseline and average across orderings.

    Ordering 1 uses the original steered_is_a assignment.
    Ordering 2 uses the swapped assignment (not steered_is_a).
    """
    steered_scores: Dict[str, Optional[float]] = {}
    baseline_scores: Dict[str, Optional[float]] = {}
    position_bias: Dict[str, Optional[float]] = {}

    for dim in DIMENSIONS:
        s_vals: List[float] = []
        b_vals: List[float] = []
        a_vals: List[float] = []   # agent_a scores per ordering (for position bias)
        b_pos: List[float] = []    # agent_b scores per ordering

        if parsed_1:
            # Ordering 1: steered_is_a -> agent_a = steered
            s_key_1 = "agent_a" if steered_is_a else "agent_b"
            b_key_1 = "agent_b" if steered_is_a else "agent_a"
            s_vals.append(parsed_1[s_key_1][dim]["score"])
            b_vals.append(parsed_1[b_key_1][dim]["score"])
            a_vals.append(parsed_1["agent_a"][dim]["score"])
            b_pos.append(parsed_1["agent_b"][dim]["score"])

        if parsed_2:
            # Ordering 2: labels swapped -> agent_a = baseline
            s_key_2 = "agent_b" if steered_is_a else "agent_a"
            b_key_2 = "agent_a" if steered_is_a else "agent_b"
            s_vals.append(parsed_2[s_key_2][dim]["score"])
            b_vals.append(parsed_2[b_key_2][dim]["score"])
            a_vals.append(parsed_2["agent_a"][dim]["score"])
            b_pos.append(parsed_2["agent_b"][dim]["score"])

        steered_scores[dim] = (
            round(sum(s_vals) / len(s_vals), 2) if s_vals else None
        )
        baseline_scores[dim] = (
            round(sum(b_vals) / len(b_vals), 2) if b_vals else None
        )
        # Position bias = mean(agent_a scores) - mean(agent_b scores)
        # Positive => judge systematically favours the A label.
        if a_vals and b_pos:
            position_bias[dim] = round(
                sum(a_vals) / len(a_vals) - sum(b_pos) / len(b_pos), 2,
            )

    return {
        "steered": steered_scores,
        "baseline": baseline_scores,
        "position_bias": position_bias,
    }


def judge_game(
    game: Dict,
    judge_name: str,
    thinking: bool = False,
    sleep_sec: float = 2.0,
) -> Dict:
    """Evaluate one game with position counterbalancing (two orderings)."""

    # Random assignment: which label the steered agent gets first
    steered_is_a = random.choice([True, False])

    # ── Ordering 1 ──
    prompt_1 = build_user_prompt(game, steered_is_a)
    raw_1, parsed_1 = _call_with_retry(
        judge_name, prompt_1, SYSTEM_PROMPT, thinking,
    )
    time.sleep(sleep_sec)

    # ── Ordering 2 (labels swapped) ──
    prompt_2 = build_user_prompt(game, not steered_is_a)
    raw_2, parsed_2 = _call_with_retry(
        judge_name, prompt_2, SYSTEM_PROMPT, thinking,
    )
    time.sleep(sleep_sec)

    # ── Raw record (for judge_scores_raw.json) ──
    raw_record = {
        "steered_is_a_ordering1": steered_is_a,
        "ordering_1": {"raw_response": raw_1, "parsed": parsed_1},
        "ordering_2": {"raw_response": raw_2, "parsed": parsed_2},
    }

    # ── Averaged processed scores ──
    processed = _average_orderings(parsed_1, parsed_2, steered_is_a)

    return {"raw": raw_record, "processed": processed}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def build_summary(processed_games: List[Dict], judge_name: str) -> Dict:
    """Aggregate per-judge mean scores across all games."""
    steered_acc = {d: [] for d in DIMENSIONS}
    baseline_acc = {d: [] for d in DIMENSIONS}
    bias_acc = {d: [] for d in DIMENSIONS}

    for g in processed_games:
        scores = g["judges"].get(judge_name)
        if not scores:
            continue
        for dim in DIMENSIONS:
            if scores["steered"].get(dim) is not None:
                steered_acc[dim].append(scores["steered"][dim])
            if scores["baseline"].get(dim) is not None:
                baseline_acc[dim].append(scores["baseline"][dim])
            if scores["position_bias"].get(dim) is not None:
                bias_acc[dim].append(scores["position_bias"][dim])

    def _mean(vals):
        return round(sum(vals) / len(vals), 3) if vals else None

    n_valid = max(
        (len(steered_acc[d]) for d in DIMENSIONS), default=0,
    )
    return {
        "n_games": n_valid,
        "steered_mean":  {d: _mean(steered_acc[d]) for d in DIMENSIONS},
        "baseline_mean": {d: _mean(baseline_acc[d]) for d in DIMENSIONS},
        "steered_overall": _mean(
            [v for d in DIMENSIONS for v in steered_acc[d]],
        ),
        "baseline_overall": _mean(
            [v for d in DIMENSIONS for v in baseline_acc[d]],
        ),
        "mean_position_bias": {d: _mean(bias_acc[d]) for d in DIMENSIONS},
    }


def print_summary(summary: Dict, judge_name: str) -> None:
    hdr = f"  JUDGE: {judge_name.upper()}"
    print(f"\n{'=' * 65}")
    print(hdr)
    print(f"  Games evaluated: {summary['n_games']}")
    print(f"{'=' * 65}")
    print(
        f"  {'Dimension':<25s} {'Steered':>8s} {'Baseline':>8s}"
        f" {'Diff':>8s} {'PosBias':>8s}"
    )
    print(f"  {'-' * 61}")
    for dim in DIMENSIONS:
        s = summary["steered_mean"].get(dim)
        b = summary["baseline_mean"].get(dim)
        pb = summary["mean_position_bias"].get(dim)
        s_s = f"{s:.2f}" if s is not None else "  n/a"
        b_s = f"{b:.2f}" if b is not None else "  n/a"
        d_s = (
            f"{s - b:+.2f}" if s is not None and b is not None else "  n/a"
        )
        pb_s = f"{pb:+.2f}" if pb is not None else "  n/a"
        print(f"  {dim:<25s} {s_s:>8s} {b_s:>8s} {d_s:>8s} {pb_s:>8s}")
    s_o = summary.get("steered_overall")
    b_o = summary.get("baseline_overall")
    print(f"  {'-' * 61}")
    if s_o is not None and b_o is not None:
        print(
            f"  {'OVERALL':<25s} {s_o:>8.2f} {b_o:>8.2f}"
            f" {s_o - b_o:>+8.2f}"
        )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM judge evaluation of negotiation transcripts.",
    )
    p.add_argument(
        "--input", default="analysis/metrics_b1_enriched.json",
        help="Path to enriched results JSON (default: analysis/metrics_b1_enriched.json)",
    )
    p.add_argument(
        "--judges", nargs="+", default=["gemini"],
        choices=["gemini", "gpt", "llama", "ollama"],
        help="Which judges to run (default: gemini)",
    )
    p.add_argument(
        "--num_games", type=int, default=None,
        help="Number of games to evaluate (default: all)",
    )
    p.add_argument(
        "--thinking", action="store_true",
        help="Enable Gemini thinking (uses 'low' instead of 'minimal')",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    p.add_argument(
        "--sleep", type=float, default=2.0,
        help="Seconds to sleep between API calls (default: 2.0)",
    )
    p.add_argument(
        "--output_raw", default="results/judge_scores_raw.json",
        help="Output path for raw responses (default: results/judge_scores_raw.json)",
    )
    p.add_argument(
        "--output", default="results/judge_scores.json",
        help="Output path for processed scores (default: results/judge_scores.json)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    # ── Validate API keys ──
    key_env = {
        "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "gpt":    ("OPENAI_API_KEY",),
        "llama":  ("GROQ_API_KEY",),
        "ollama": (),  # local, no API key needed
    }
    for j in args.judges:
        keys = key_env[j]
        if keys and not any(os.environ.get(k) for k in keys):
            print(
                f"[ERROR] No API key for '{j}'. "
                f"Set one of: {', '.join(keys)}"
            )
            sys.exit(1)

    # ── Load data ──
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)
    with open(input_path) as f:
        data = json.load(f)

    games = data["games"]
    if args.num_games is not None:
        games = games[: args.num_games]

    n = len(games)
    print(f"Loaded {n} games from {input_path}")
    print(f"Judges: {', '.join(args.judges)}")
    print(f"Thinking: {'enabled' if args.thinking else 'disabled (minimal)'}")
    print()

    # ── Run evaluation ──
    all_raw: List[Dict] = []      # full records (raw + processed)
    all_processed: List[Dict] = []  # stripped to processed only

    for i, game in enumerate(games):
        game_meta = {
            "game_id":      game.get("game_id", i),
            "title":        game["title"],
            "category":     game["category"],
            "steered_role": game["steered_role"],
            "advantage":    game.get("advantage"),
            "agreed":       game["agreed"],
            "agreed_price": game.get("agreed_price"),
        }
        raw_game = {**game_meta, "judges": {}}
        proc_game = {**game_meta, "judges": {}}

        for judge_name in args.judges:
            print(
                f"Game {i + 1}/{n} | {judge_name} | "
                f"{game['category']} | {game['title'][:40]}"
            )
            result = judge_game(
                game, judge_name,
                thinking=args.thinking,
                sleep_sec=args.sleep,
            )
            raw_game["judges"][judge_name] = result["raw"]
            proc_game["judges"][judge_name] = result["processed"]

            # Quick inline summary
            p = result["processed"]
            if p["steered"].get("firmness") is not None:
                sf = p["steered"]["firmness"]
                bf = p["baseline"]["firmness"]
                print(f"    firmness: steered={sf:.1f}  baseline={bf:.1f}")

        all_raw.append(raw_game)
        all_processed.append(proc_game)

    # ── Save raw ──
    raw_path = Path(args.output_raw)
    with open(raw_path, "w") as f:
        json.dump({"games": all_raw}, f, indent=2, ensure_ascii=False)
    print(f"\nRaw scores saved to {raw_path}")

    # ── Save processed + summary ──
    processed_out: Dict = {"games": all_processed, "summary": {}}
    for judge_name in args.judges:
        summary = build_summary(all_processed, judge_name)
        processed_out["summary"][judge_name] = summary
        print_summary(summary, judge_name)

    proc_path = Path(args.output)
    with open(proc_path, "w") as f:
        json.dump(processed_out, f, indent=2, ensure_ascii=False)
    print(f"Processed scores saved to {proc_path}")


if __name__ == "__main__":
    main()
