import sys

# flake8: noqa: E402
sys.path.append(".")

import unittest

from brain.higher_cognitives.decision_making import DecisionMaking


class TestDecisionMaking(unittest.TestCase):
    def setUp(self):
        self.decision_making = DecisionMaking()

    def test_decision_making_execution(self):
        counts = self.decision_making.execute()
        self.assertIsInstance(counts, dict)

    def test_decision_making_execution_counts(self):
        counts = self.decision_making.execute()
        self.assertIn("000", counts)
        self.assertIn("111", counts)
        self.assertGreater(counts["000"], 0)
        self.assertGreater(counts["111"], 0)


if __name__ == "__main__":
    unittest.main()
