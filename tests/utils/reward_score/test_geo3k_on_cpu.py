# Copyright 2026 verl contributors
# SPDX-License-Identifier: Apache-2.0

import pytest

from verl.utils.reward_score.geo3k import compute_score, format_reward


@pytest.mark.parametrize("tag", ["think", "mm:think"])
def test_correct_answer_receives_accuracy_and_format_reward(tag):
    answer = f"<{tag}>reasoning</{tag}>" + r"\boxed{2}"
    assert format_reward(answer) == 1.0
    assert compute_score(answer, "2") == 1.0


@pytest.mark.parametrize(
    "answer",
    [
        r"<mm:think>reasoning</think>\boxed{2}",
        r"<think>reasoning</mm:think>\boxed{2}",
        r"<mm:think>reasoning\boxed{2}",
        r"\boxed{2}",
    ],
)
def test_malformed_think_tags_do_not_receive_format_reward(answer):
    assert format_reward(answer) == 0.0
    assert compute_score(answer, "2") == 0.9
