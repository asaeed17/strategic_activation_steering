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
    build_player_system,
    extract_behavioral_metrics,
    RESOURCE_CONFIGS,
    MAX_ROUNDS,
    MAX_PROPOSALS,
)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class TestScoring:
    def test_total_resources(self):
        assert compute_score({"X": 10, "Y": 20}) == 30

    def test_zero_resources(self):
        assert compute_score({"X": 0, "Y": 0}) == 0

    def test_single_resource(self):
        assert compute_score({"X": 15, "Y": 0}) == 15


class TestBalance:
    def test_perfect_balance(self):
        assert compute_balance_ratio({"X": 10, "Y": 10}) == 1.0

    def test_half_balance(self):
        assert compute_balance_ratio({"X": 5, "Y": 10}) == 0.5

    def test_zero_balance(self):
        assert compute_balance_ratio({"X": 0, "Y": 10}) == 0.0

    def test_all_zero(self):
        assert compute_balance_ratio({"X": 0, "Y": 0}) == 0.0


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

class TestParsing:
    def test_equal_trade(self):
        result = parse_action("I'll trade some resources. TRADE: GIVE 5 X, RECEIVE 5 Y")
        assert result == {
            "type": "propose",
            "give_qty": 5, "give_res": "X",
            "receive_qty": 5, "receive_res": "Y",
        }

    def test_unequal_trade(self):
        result = parse_action("TRADE: GIVE 10 X, RECEIVE 15 Y")
        assert result["give_qty"] == 10
        assert result["receive_qty"] == 15

    def test_trade_lowercase(self):
        result = parse_action("trade: give 3 y, receive 3 x")
        assert result is not None
        assert result["type"] == "propose"
        assert result["give_res"] == "Y"
        assert result["receive_res"] == "X"

    def test_same_resource_fails(self):
        assert parse_action("TRADE: GIVE 5 X, RECEIVE 5 X") is None

    def test_accept(self):
        result = parse_action("That sounds fair. ACCEPT")
        assert result["type"] == "accept"

    def test_none(self):
        result = parse_action("I'll pass this turn. NONE")
        assert result["type"] == "none"

    def test_no_action(self):
        assert parse_action("Just some random text without any action keywords.") is None

    def test_last_action_wins(self):
        result = parse_action("NONE... actually wait, ACCEPT")
        assert result["type"] == "accept"

    def test_trade_takes_priority(self):
        result = parse_action("TRADE: GIVE 3 X, RECEIVE 2 Y and I ACCEPT")
        assert result["type"] == "propose"

    def test_lopsided_trade(self):
        result = parse_action("TRADE: GIVE 3 X, RECEIVE 10 Y")
        assert result["give_qty"] == 3
        assert result["receive_qty"] == 10

    def test_reject_not_recognised(self):
        """REJECT is not a valid action in NegotiationArena-style game."""
        result = parse_action("REJECT")
        assert result is None

    def test_end_not_recognised(self):
        """END is not a valid action in NegotiationArena-style game."""
        result = parse_action("END")
        assert result is None


# ---------------------------------------------------------------------------
# Trade Validation
# ---------------------------------------------------------------------------

class TestTradeValidation:
    def test_valid_trade(self):
        proposer = {"X": 10, "Y": 5}
        responder = {"X": 5, "Y": 10}
        trade = {"type": "propose", "give_qty": 3, "give_res": "X",
                 "receive_qty": 3, "receive_res": "Y"}
        assert validate_trade(trade, proposer, responder) is True

    def test_valid_unequal_trade(self):
        proposer = {"X": 10, "Y": 5}
        responder = {"X": 5, "Y": 10}
        trade = {"type": "propose", "give_qty": 3, "give_res": "X",
                 "receive_qty": 8, "receive_res": "Y"}
        assert validate_trade(trade, proposer, responder) is True

    def test_insufficient_proposer_resources(self):
        proposer = {"X": 2, "Y": 5}
        responder = {"X": 5, "Y": 10}
        trade = {"type": "propose", "give_qty": 5, "give_res": "X",
                 "receive_qty": 3, "receive_res": "Y"}
        assert validate_trade(trade, proposer, responder) is False

    def test_insufficient_responder_resources(self):
        proposer = {"X": 10, "Y": 5}
        responder = {"X": 5, "Y": 2}
        trade = {"type": "propose", "give_qty": 3, "give_res": "X",
                 "receive_qty": 5, "receive_res": "Y"}
        assert validate_trade(trade, proposer, responder) is False

    def test_zero_quantity(self):
        proposer = {"X": 10, "Y": 5}
        responder = {"X": 5, "Y": 10}
        trade = {"type": "propose", "give_qty": 0, "give_res": "X",
                 "receive_qty": 3, "receive_res": "Y"}
        assert validate_trade(trade, proposer, responder) is False

    def test_non_propose_action(self):
        assert validate_trade({"type": "accept"}, {"X": 10}, {"Y": 10}) is False


# ---------------------------------------------------------------------------
# Trade Execution
# ---------------------------------------------------------------------------

class TestTradeExecution:
    def test_equal_trade(self):
        p, r = execute_trade(
            {"X": 10, "Y": 5}, {"X": 5, "Y": 10},
            {"type": "propose", "give_qty": 3, "give_res": "X",
             "receive_qty": 3, "receive_res": "Y"},
        )
        assert p == {"X": 7, "Y": 8}
        assert r == {"X": 8, "Y": 7}

    def test_unequal_trade_proposer_gains(self):
        p, r = execute_trade(
            {"X": 10, "Y": 5}, {"X": 5, "Y": 10},
            {"type": "propose", "give_qty": 3, "give_res": "X",
             "receive_qty": 5, "receive_res": "Y"},
        )
        assert p == {"X": 7, "Y": 10}
        assert r == {"X": 8, "Y": 5}
        assert compute_score(p) == 17
        assert compute_score(r) == 13

    def test_unequal_trade_proposer_loses(self):
        p, r = execute_trade(
            {"X": 10, "Y": 5}, {"X": 5, "Y": 10},
            {"type": "propose", "give_qty": 5, "give_res": "X",
             "receive_qty": 3, "receive_res": "Y"},
        )
        assert p == {"X": 5, "Y": 8}
        assert r == {"X": 10, "Y": 7}

    def test_does_not_mutate_originals(self):
        orig_p = {"X": 10, "Y": 5}
        orig_r = {"X": 5, "Y": 10}
        execute_trade(
            orig_p, orig_r,
            {"type": "propose", "give_qty": 3, "give_res": "X",
             "receive_qty": 5, "receive_res": "Y"},
        )
        assert orig_p == {"X": 10, "Y": 5}
        assert orig_r == {"X": 5, "Y": 10}

    def test_total_resources_conserved(self):
        p, r = execute_trade(
            {"X": 20, "Y": 5}, {"X": 5, "Y": 20},
            {"type": "propose", "give_qty": 8, "give_res": "X",
             "receive_qty": 12, "receive_res": "Y"},
        )
        total_before = 20 + 5 + 5 + 20
        total_after = sum(p.values()) + sum(r.values())
        assert total_before == total_after


# ---------------------------------------------------------------------------
# Rule-Based Opponent
# ---------------------------------------------------------------------------

class TestRuleBasedOpponent:
    def test_accepts_beneficial_trade(self):
        own = {"X": 5, "Y": 25}
        opp = {"X": 25, "Y": 5}
        proposal = {"type": "propose", "give_qty": 5, "give_res": "X",
                     "receive_qty": 3, "receive_res": "Y"}
        action = rule_based_action(own, opp, proposal, turn=1)
        assert action["type"] == "accept"

    def test_accepts_equal_trade(self):
        own = {"X": 5, "Y": 25}
        opp = {"X": 25, "Y": 5}
        proposal = {"type": "propose", "give_qty": 5, "give_res": "X",
                     "receive_qty": 5, "receive_res": "Y"}
        action = rule_based_action(own, opp, proposal, turn=1)
        assert action["type"] == "accept"

    def test_rejects_harmful_trade_with_none(self):
        own = {"X": 5, "Y": 25}
        opp = {"X": 25, "Y": 5}
        proposal = {"type": "propose", "give_qty": 3, "give_res": "X",
                     "receive_qty": 5, "receive_res": "Y"}
        action = rule_based_action(own, opp, proposal, turn=1)
        assert action["type"] == "none"

    def test_proposes_lopsided_trade(self):
        own = {"X": 25, "Y": 5}
        opp = {"X": 5, "Y": 25}
        action = rule_based_action(own, opp, None, turn=0)
        assert action["type"] == "propose"
        assert action["give_res"] == "X"
        assert action["receive_res"] == "Y"
        assert action["receive_qty"] >= action["give_qty"]

    def test_none_when_balanced(self):
        own = {"X": 15, "Y": 15}
        opp = {"X": 15, "Y": 15}
        action = rule_based_action(own, opp, None, turn=0)
        assert action["type"] == "none"


# ---------------------------------------------------------------------------
# Private Information
# ---------------------------------------------------------------------------

class TestPrivateInfo:
    def test_prompt_does_not_contain_opponent_resources(self):
        prompt = build_player_system(1, {"X": 25, "Y": 5})
        assert "no knowledge" in prompt.lower()
        assert "25 X and 5 Y" in prompt  # own resources shown
        # Should NOT contain opponent's resources
        assert "5 X and 25 Y" not in prompt

    def test_prompt_allows_voluntary_disclosure(self):
        prompt = build_player_system(1, {"X": 25, "Y": 5})
        assert "disclose" in prompt.lower()


# ---------------------------------------------------------------------------
# Full Game
# ---------------------------------------------------------------------------

class TestFullGame:
    def test_rulebased_both_sides(self):
        result = run_single_exchange_game(
            config=(25, 5, 5, 25),
            steered_player=None,
            generate_fn=None,
            rulebased=True,
        )
        assert result["parse_errors"] == 0
        total_initial = sum(result["p1_initial"].values()) + sum(result["p2_initial"].values())
        total_final = sum(result["p1_final"].values()) + sum(result["p2_final"].values())
        assert total_initial == total_final

    def test_various_configs(self):
        for config in RESOURCE_CONFIGS[:10]:
            result = run_single_exchange_game(
                config=config,
                steered_player=None,
                generate_fn=None,
                rulebased=True,
            )
            assert result["parse_errors"] == 0
            total_initial = (
                sum(result["p1_initial"].values())
                + sum(result["p2_initial"].values())
            )
            total_final = (
                sum(result["p1_final"].values())
                + sum(result["p2_final"].values())
            )
            assert total_initial == total_final

    def test_game_length_within_bounds(self):
        result = run_single_exchange_game(
            config=(25, 5, 5, 25),
            steered_player=None,
            generate_fn=None,
            rulebased=True,
        )
        assert result["game_length"] <= MAX_ROUNDS


# ---------------------------------------------------------------------------
# Behavioral Metrics
# ---------------------------------------------------------------------------

class TestBehavioralMetrics:
    def test_word_count(self):
        m = extract_behavioral_metrics("I want to trade five resources")
        assert m["word_count"] == 6

    def test_hedge_count(self):
        m = extract_behavioral_metrics("Maybe I could possibly trade")
        assert m["hedge_count"] >= 2

    def test_fairness_count(self):
        m = extract_behavioral_metrics("This is a fair and balanced trade for both of us")
        assert m["fairness_count"] >= 2


# ---------------------------------------------------------------------------
# Resource Configs
# ---------------------------------------------------------------------------

class TestResourceConfigs:
    def test_all_configs_have_positive_resources(self):
        for config in RESOURCE_CONFIGS:
            assert all(v > 0 for v in config), f"Config has non-positive: {config}"

    def test_all_configs_are_4_tuples(self):
        for config in RESOURCE_CONFIGS:
            assert len(config) == 4, f"Config not length 4: {config}"

    def test_minimum_configs(self):
        assert len(RESOURCE_CONFIGS) >= 50


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_max_rounds(self):
        assert MAX_ROUNDS == 8

    def test_max_proposals(self):
        assert MAX_PROPOSALS == 3
