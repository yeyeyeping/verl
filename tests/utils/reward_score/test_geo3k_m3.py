"""CPU-only regressions for Geometry3k M3/adaptive output formats."""

import importlib.util
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[3] / "verl/utils/reward_score/geo3k.py"
spec = importlib.util.spec_from_file_location("geo3k_under_test", SOURCE)
geo3k = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geo3k)


class GeometryFormatTest(unittest.TestCase):
    def test_formats(self):
        cases = [
            (r"<think>reason</think>\boxed{42}", 1),
            (r"<mm:think>reason</mm:think>\boxed{42}", 1),
            (" \n<mm:think>reason\n</mm:think>\\boxed{42}\n ", 1),
            (r"\boxed{42}", 1),
            (r"The answer is \boxed{42}.", 1),
            (r"</mm:think>\boxed{42}", 1),
            (r"<mm:think>reason</mm:think>\boxed{\frac{1}{2}}", 1),
            (r"<mm:think>reason</think>\boxed{42}", 0),
            (r"<think>reason</mm:think>\boxed{42}", 0),
            (r"<mm:think>reason \boxed{42}", 0),
            (r"<mm:think>\boxed{42}</mm:think>done", 0),
            (r"<mm:think><think>x</think></mm:think>\boxed{42}", 0),
            (r"<think>x</think><think>y</think>\boxed{42}", 0),
            (r"</think>\boxed{42}", 0),
            (r"</mm:think></mm:think>\boxed{42}", 0),
            (r"\boxed{42", 0),
            (r"\boxed{}", 0),
            ("", 0),
            ("42", 0),
        ]
        for response, expected in cases:
            with self.subTest(response=response):
                self.assertEqual(geo3k.format_reward(response), expected)

    def test_reward_weights_and_accuracy(self):
        for tag in ("think", "mm:think"):
            self.assertEqual(geo3k.compute_score(f"<{tag}>reason</{tag}>" + r"\boxed{42}", "42"), 1.0)
            self.assertEqual(geo3k.compute_score(f"<{tag}>reason</{tag}>" + r"\boxed{41}", "42"), 0.1)
        self.assertEqual(geo3k.compute_score(r"\boxed{42}", "42"), 1.0)
        self.assertEqual(geo3k.compute_score(r"<mm:think>x</think>\boxed{42}", "42"), 0.9)
        self.assertEqual(geo3k.compute_score(r"<mm:think>x</think>\boxed{41}", "42"), 0.0)


if __name__ == "__main__":
    unittest.main()
