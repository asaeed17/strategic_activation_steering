"""Tests for resource_exchange_game.py game logic (no GPU needed)."""

import pytest
from resource_exchange_game import (
    compute_score,
    compute_balance_ratio,
    parse_action,
    validate_trade,
    execute_trade,
    rule_based_action,
    run_single_exchange_game,
    summarise,
    build_player_system,
    build_turn_prompt,
    extract_behavioral_metrics,
    extract_message,
    RESOURCE_CONFIGS,
    MAX_ROUNDS,
    MAX_PROPOSALS,
)


class TestScoring:
    def test_total_resources(self):
        assert compute_score({"X": 10, "Y": 20}) == 30

    def test_zero(self):
        assert compute_score({"X": 0, "Y": 0}) == 0


class TestBalance:
    def test_perfect(self):
        assert compute_balance_ratio({"X": 10, "Y": 10}) == 1.0

    def test_half(self):
        assert compute_balance_ratio({"X": 5, "Y": 10}) == 0.5

    def test_zero(self):
        assert compute_balance_ratio({"X": 0, "Y": 10}) == 0.0


class TestParsing:
    def test_equal_trade(self):
        r = parse_action("TRADE: GIVE 5 X, RECEIVE 5 Y")
        assert r == {"type": "propose", "give_qty": 5, "give_res": "X",
                     "receive_qty": 5, "receive_res": "Y"}

    def test_unequal_trade(self):
        r = parse_action("TRADE: GIVE 10 X, RECEIVE 15 Y")
        assert r["give_qty"] == 10 and r["receive_qty"] == 15

    def test_same_resource_fails(self):
        assert parse_action("TRADE: GIVE 5 X, RECEIVE 5 X") is None

    def test_accept(self):
        assert parse_action("ACCEPT")["type"] == "accept"

    def test_none(self):
        assert parse_action("NONE")["type"] == "none"

    def test_no_action(self):
        assert parse_action("random text") is None

    def test_reject_not_recognised(self):
        assert parse_action("REJECT") is None

    def test_end_not_recognised(self):
        assert parse_action("END") is None

    def test_trade_takes_priority(self):
        assert parse_action("TRADE: GIVE 3 X, RECEIVE 2 Y ACCEPT")["type"] == "propose"


class TestTradeValidation:
    def test_valid(self):
        t = {"type": "propose", "give_qty": 3, "give_res": "X",
             "receive_qty": 3, "receive_res": "Y"}
        assert validate_trade(t, {"X": 10, "Y": 5}, {"X": 5, "Y": 10})

    def test_insufficient_proposer(self):
        t = {"type": "propose", "give_qty": 15, "give_res": "X",
             "receive_qty": 3, "receive_res": "Y"}
        assert not validate_trade(t, {"X": 10, "Y": 5}, {"X": 5, "Y": 10})

    def test_insufficient_responder(self):
        t = {"type": "propose", "give_qty": 3, "give_res": "X",
             "receive_qty": 15, "receive_res": "Y"}
        assert not validate_trade(t, {"X": 10, "Y": 5}, {"X": 5, "Y": 10})


class TestTradeExecution:
    def test_unequal(self):
        p, r = execute_trade({"X": 10, "Y": 5}, {"X": 5, "Y": 10},
                             {"type": "propose", "give_qty": 3, "give_res": "X",
                              "receive_qty": 5, "receive_res": "Y"})
        assert p == {"X": 7, "Y": 10}
        assert r == {"X": 8, "Y": 5}

    def test_conserved(self):
        p, r = execute_trade({"X": 20, "Y": 5}, {"X": 5, "Y": 20},
                             {"type": "propose", "give_qty": 8, "give_res": "X",
                              "receive_qty": 12, "receive_res": "Y"})
        assert sum(p.values()) + sum(r.values()) == 50

    def test_no_mutation(self):
        orig = {"X": 10, "Y": 5}
        execute_trade(orig, {"X": 5, "Y": 10},
                      {"type": "propose", "give_qty": 3, "give_res": "X",
                       "receive_qty": 5, "receive_res": "Y"})
        assert orig == {"X": 10, "Y": 5}


class TestRuleBasedOpponent:
    def test_accepts_beneficial(self):
        p = {"type": "propose", "give_qty": 5, "give_res": "X", "receive_qty": 3, "receive_res": "Y"}
        assert rule_based_action({"X": 5, "Y": 25}, {"X": 25, "Y": 5}, p, 1)["type"] == "accept"

    def test_rejects_with_none(self):
        p = {"type": "propose", "give_qty": 3, "give_res": "X", "receive_qty": 5, "receive_res": "Y"}
        assert rule_based_action({"X": 5, "Y": 25}, {"X": 25, "Y": 5}, p, 1)["type"] == "none"

    def test_proposes_when_imbalanced(self):
        assert rule_based_action({"X": 25, "Y": 5}, {"X": 5, "Y": 25}, None, 0)["type"] == "propose"

    def test_none_when_balanced(self):
        assert rule_based_action({"X": 15, "Y": 15}, {"X": 15, "Y": 15}, None, 0)["type"] == "none"


class TestPrivateInfo:
    def test_no_opponent_in_prompt(self):
        prompt = build_player_system(1, {"X": 25, "Y": 5})
        assert "no knowledge" in prompt.lower()
        assert "5 X and 25 Y" not in prompt


class TestFullGame:
    def test_rulebased(self):
        result = run_single_exchange_game(
            config=(25, 5, 5, 25), steered_player=None,
            generate_fn=None, rulebased=True)
        assert result["parse_errors"] == 0

    def test_various_configs(self):
        for config in RESOURCE_CONFIGS[:5]:
            result = run_single_exchange_game(
                config=config, steered_player=None,
                generate_fn=None, rulebased=True)
            assert result["parse_errors"] == 0


class TestSummarise:
    def test_basic(self):
        games = [run_single_exchange_game(
            config=c, steered_player=None, generate_fn=None, rulebased=True)
            for c in RESOURCE_CONFIGS[:3]]
        s = summarise(games, player=1)
        assert s["n_games"] == 3
        assert "mean_score" in s


class TestExtractMessage:
    def test_strips_trade_line(self):
        text = "I think we should trade fairly.\nTRADE: GIVE 5 X, RECEIVE 5 Y"
        msg = extract_message(text)
        assert "I think we should trade fairly" in msg
        assert "TRADE" not in msg

    def test_strips_accept(self):
        text = "That seems reasonable to me.\nACCEPT"
        msg = extract_message(text)
        assert "reasonable" in msg
        assert "ACCEPT" not in msg

    def test_strips_none(self):
        text = "I'll wait and see.\nNONE"
        msg = extract_message(text)
        assert "wait" in msg
        assert "NONE" not in msg

    def test_single_line_fallback(self):
        text = "TRADE: GIVE 5 X, RECEIVE 5 Y"
        msg = extract_message(text)
        assert len(msg) > 0

    def test_multiline_reasoning(self):
        text = "Let me think about this.\nI want more Y.\nTRADE: GIVE 3 X, RECEIVE 5 Y"
        msg = extract_message(text)
        assert "Let me think" in msg
        assert "I want more Y" in msg
        assert "TRADE" not in msg


class TestTurnPromptMessages:
    def test_opponent_message_included(self):
        prompt = build_turn_prompt(
            1, {"X": 25, "Y": 5}, 1,
            opponent_message="I have plenty of X to share",
        )
        assert "I have plenty of X to share" in prompt
        assert "BLUE says" in prompt

    def test_no_opponent_message(self):
        prompt = build_turn_prompt(1, {"X": 25, "Y": 5}, 0)
        assert "says" not in prompt

    def test_message_with_proposal(self):
        proposal = {"type": "propose", "give_qty": 5, "give_res": "X",
                     "receive_qty": 3, "receive_res": "Y"}
        prompt = build_turn_prompt(
            1, {"X": 25, "Y": 5}, 1,
            opponent_message="How about a trade?",
            opponent_action=proposal,
        )
        assert "How about a trade?" in prompt
        assert "proposes" in prompt


class TestMessagePassing:
    def test_messages_in_transcript(self):
        result = run_single_exchange_game(
            config=(25, 5, 5, 25), steered_player=None,
            generate_fn=None, rulebased=True)
        for turn_record in result["transcript"]:
            assert "message" in turn_record
            assert isinstance(turn_record["message"], str)
            assert len(turn_record["message"]) > 0


class TestSystemPrompt:
    def test_proposal_limit_in_prompt(self):
        prompt = build_player_system(1, {"X": 25, "Y": 5})
        assert "at most 3 proposals" in prompt

    def test_message_framing(self):
        prompt = build_player_system(1, {"X": 25, "Y": 5})
        assert "message to the other player" in prompt


class TestConstants:
    def test_max_rounds(self):
        assert MAX_ROUNDS == 8

    def test_max_proposals(self):
        assert MAX_PROPOSALS == 3

    def test_min_configs(self):
        assert len(RESOURCE_CONFIGS) >= 50
