import asyncio
import sys
sys.path.append('.')
sys.path.append('../')
import growingnn as gnn
import unittest
import pandas as pd
import random
from unittest.mock import MagicMock

class TestSimulationScore(unittest.TestCase):

    def setUp(self):
        self.global_history_mock = MagicMock()
        self.history_mock = MagicMock()

    def test_new_max_loss(self):
        self.global_history_mock.Y = {'loss': [0.3, 0.5, 0.2]}
        sim_score = gnn.Simulation_score(gnn.Simulation_score.LOSS)
        sim_score.new_max_loss(self.global_history_mock)
        self.assertEqual(sim_score.max_loss, 0.5)

    def test_grade_accuracy_mode(self):
        sim_score = gnn.Simulation_score(gnn.Simulation_score.ACCURACY)
        result = sim_score.grade(0.9, self.history_mock)
        self.assertEqual(result, 0.9)

    def test_grade_loss_mode(self):
        sim_score = gnn.Simulation_score(gnn.Simulation_score.LOSS)
        sim_score.max_loss = 0.5
        self.history_mock.get_last.return_value = 0.3
        result = sim_score.grade(0.9, self.history_mock)
        self.assertEqual(result, 0.2)

    def test_grade_loss_mode_min_threshold(self):
        sim_score = gnn.Simulation_score(gnn.Simulation_score.LOSS)
        sim_score.max_loss = 0.1
        self.history_mock.get_last.return_value = 0.2
        result = sim_score.grade(0.9, self.history_mock)
        self.assertEqual(result, 1.e-17)

    def test_grade_accuracy(self):
        score = gnn.Simulation_score(mode=gnn.Simulation_score.ACCURACY)
        result = score.grade(acc=0.9, history=None)
        self.assertEqual(result, 0.9)
        
    def test_grade_loss(self):
        score = gnn.Simulation_score(mode=gnn.Simulation_score.LOSS)
        global_history = MagicMock()
        global_history.Y = {'loss': [0.5, 0.4, 0.3]}
        score.new_max_loss(global_history)
        history = MagicMock()
        history.get_last.return_value = 0.3
        result = score.grade(acc=None, history=history)
        expected = 0.5 - 0.3
        self.assertAlmostEqual(result, expected)

if __name__ == '__main__':
    unittest.main()
