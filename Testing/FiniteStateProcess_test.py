import unittest

from werkzeug.datastructures import MultiDict

from AQTSMetrics import SPA
from ApproxBisimulation import ProbabilisticFiniteStateProcess  # Assuming the class is in 'finite_state_process.py'


class TestFiniteStateProcess(unittest.TestCase):

    def setUp(self):
        self.FSP1_states = {"s1", "s2", "s3"}
        self.FSP1_start_state = "s1"
        self.FSP1_transition_function = {
            "s1": MultiDict([("a", {"s2": 1}), ("b", {"s3": 1})]),
            "s2": MultiDict([("c", {"s3": 1})])
        }
        self.FSP1 = ProbabilisticFiniteStateProcess(
            states=self.FSP1_states,
            start_state=self.FSP1_start_state,
            transition_function=self.FSP1_transition_function
        )

        self.FSP2 = ProbabilisticFiniteStateProcess(
            states={"s1", "s2", "s3"},
            start_state="s1",
            transition_function= {
                "s1": MultiDict([("a", {"s2": 0.5, "s3": 0.5}), ("b", {"s3": 1})]),
                "s2": MultiDict([("c", {"s1": 1/3, "s2": 1/3, "s3": 1/3})])
            }
        )

    def test_initialization(self):
        self.assertEqual(self.FSP1.start, self.FSP1_start_state)
        self.assertEqual(self.FSP1.states, self.FSP1_states)
        self.assertEqual(self.FSP1.transition_function, self.FSP1_transition_function)

    def test_target_states(self):
        self.assertEqual(self.FSP1.target_states("s1", "a")[0], {"s2": 1})
        self.assertEqual(self.FSP1.target_states("s1", "b")[0], {"s3": 1})
        self.assertEqual(self.FSP1.target_states("s2", "c")[0], {"s3": 1})
        self.assertEqual(self.FSP1.target_states("s2", "d"), list())  # Non-existent action

        with self.assertRaises(ValueError):
            self.FSP1.target_states("s4", "a")  # Non-existent state

    def test_extension(self):
        expected_extension = {"s1": {"a", "b"}, "s2": {"c"}, "s3": set()}
        self.assertEqual(expected_extension, self.FSP1.extension())

    def test_all_actions(self):
        self.assertEqual(self.FSP1.all_actions(), {"a", "b", "c", ProbabilisticFiniteStateProcess.TAU})

    def test_actions(self):
        self.assertEqual(self.FSP1.actions("s1"), ["a", "b"])
        self.assertEqual(self.FSP1.actions("s2"), ["c"])
        self.assertEqual(self.FSP1.actions("s3"), list())

        with self.assertRaises(ValueError):
            self.FSP1.actions("s4")  # Non-existent state

    def test_is_tau(self):
        self.assertTrue(self.FSP1.is_tau("τ"))
        self.assertFalse(self.FSP1.is_tau("a"))

    def test_add_state(self):
        self.FSP1.add_state("s4")
        self.assertIn("s4", self.FSP1.states)

    def test_add_transition(self):
        self.FSP1.add_distribution("s1", "d", {"s3": 0})
        self.assertEqual(self.FSP1.target_states("s1", "d")[0], {"s3": 0})
        self.assertEqual(self.FSP1.actions("s1"), ["a", "b", "d"])

        with self.assertRaises(ValueError):
            self.FSP1.add_distribution("s4", "e", {"s5": 1.0})  # Non-existent states

    def test_to_spa(self):
        spa_model = self.FSP1.to_spa()
        self.assertIsInstance(spa_model, SPA)
        self.assertIn("s1", spa_model.data)
        self.assertIn("a", spa_model.data["s1"])
        self.assertAlmostEqual(spa_model.data["s1"]["a"][0]["s2"], 1.0)  # Single transition probability

    def test_to_spa2(self):
        spa_model = self.FSP2.to_spa()
        self.assertIsInstance(spa_model, SPA)
        self.assertIn("s1", spa_model.data)
        self.assertIn("a", spa_model.data["s1"])
        self.assertEqual(spa_model.data["s1"]["a"][0], {"s2": 0.5, "s3": 0.5})  # Multiple transition probabilities
        self.assertEqual(spa_model.data["s2"]["c"][0], {"s1": 1/3, "s2": 1/3, "s3": 1/3})  # Multiple transition probabilities

    def test_repr(self):
        repr_str = repr(self.FSP1)
        self.assertIn("FiniteStateProcess", repr_str)
        self.assertIn("s1", repr_str)