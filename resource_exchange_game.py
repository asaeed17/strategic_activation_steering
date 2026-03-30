#!/usr/bin/env python3
"""
resource_exchange_game.py

Resource Exchange Game with activation steering, based on NegotiationArena
(Bianchi et al., ICML 2024). Two players with complementary asymmetric
resource bundles (X and Y) trade over multiple rounds to maximise total
resources.

This serves as cross-task validation for steering effects found in the
Ultimatum Game: if dose-response and layer-location findings replicate
on a structurally different (multi-turn, multi-issue) game, the claims
are more generalisable.

Usage:
    # Self-test (no GPU needed)
    python resource_exchange_game.py --test

    # LLM-vs-LLM with steering (GPU required)
    python resource_exchange_game.py --model qwen2.5-7b --dimension firmness \\
        --layers 10 --alpha 7 --steered_player 1 --n_games 50 --paired \\
        --vectors_dir vectors/ultimatum_10dim_20pairs_general_matched/negotiation \\
        --output_dir results/resource_exchange/firmness_L10_a7

    # Rule-based opponent (cheaper gridsearch)
    python resource_exchange_game.py --model qwen2.5-7b --dimension firmness \\
        --layers 10 --alpha 7 --steered_player 1 --n_games 50 --paired \\
        --rulebased --output_dir results/resource_exchange/firmness_L10_a7_rb
"""

import re
import json
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
