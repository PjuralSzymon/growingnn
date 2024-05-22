import asyncio
import sys
sys.path.append('.')
sys.path.append('../')
import growingnn as gnn
import unittest
import pandas as pd
import random
from unittest.mock import MagicMock

class TestSimulationScheduler(unittest.TestCase):

    def setUp(self):
        self.hist_detail_mock = MagicMock()
        self.hist_detail_mock.Y = {
            'iteration_acc_train': [],
            'accuracy': []
        }

    def test_constant_mode_simulation(self):
        scheduler = gnn.SimulationScheduler(gnn.SimulationScheduler.CONSTANT, 100, 10)
        result = scheduler.can_simulate(1, self.hist_detail_mock)
        self.assertTrue(result)

    def test_progress_check_simulation_no_progress(self):
        scheduler = gnn.SimulationScheduler(gnn.SimulationScheduler.PROGRESS_CHECK, 100, 10)
        self.hist_detail_mock.Y['iteration_acc_train'] = [0.9, 0.9]
        result = scheduler.can_simulate(1, self.hist_detail_mock)
        self.assertTrue(result)

    def test_progress_check_simulation_with_progress(self):
        scheduler = gnn.SimulationScheduler(gnn.SimulationScheduler.PROGRESS_CHECK, 100, 10)
        self.hist_detail_mock.Y['iteration_acc_train'] = [0.8, 0.9]
        result = scheduler.can_simulate(1, self.hist_detail_mock)
        self.assertFalse(result)

    def test_overfit_check_simulation_overfit_detected(self):
        scheduler = gnn.SimulationScheduler(gnn.SimulationScheduler.OVERFIT_CHECK, 100, 10)
        self.hist_detail_mock.learning_capable.return_value = True
        result = scheduler.can_simulate(1, self.hist_detail_mock)
        self.assertTrue(result)

    def test_overfit_check_simulation_no_overfit(self):
        scheduler = gnn.SimulationScheduler(gnn.SimulationScheduler.OVERFIT_CHECK, 100, 10)
        self.hist_detail_mock.learning_capable.return_value = False
        result = scheduler.can_simulate(1, self.hist_detail_mock)
        self.assertFalse(result)

    def test_get_mode_label(self):
        scheduler = gnn.SimulationScheduler(gnn.SimulationScheduler.CONSTANT, 100, 10)
        self.assertEqual(scheduler.get_mode_label(), "simconstant")
        scheduler = gnn.SimulationScheduler(gnn.SimulationScheduler.PROGRESS_CHECK, 100, 10)
        self.assertEqual(scheduler.get_mode_label(), "simprogres")
        scheduler = gnn.SimulationScheduler(gnn.SimulationScheduler.OVERFIT_CHECK, 100, 10)
        self.assertEqual(scheduler.get_mode_label(), "simoverfit")

if __name__ == '__main__':
    unittest.main()
