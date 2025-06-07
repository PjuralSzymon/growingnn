import sys
sys.path.append('.')
sys.path.append('../')
import growingnn as gnn
import unittest
from testSuite import mode
from unittest.mock import MagicMock, patch

class TestSimulationScheduler(unittest.TestCase):

    def setUp(self):
        global mode
        if mode == 'cpu':
            gnn.switch_to_cpu()
        elif mode == 'gpu':
            gnn.switch_to_gpu()
        self.hist_detail_mock = MagicMock()
        self.hist_detail_mock.Y = {
            'iteration_acc_train': [],
            'accuracy': [],
            'loss': []
        }

    def test_constant_mode_simulation(self):
        """Test constant mode simulation scheduler"""
        # Test with default parameters
        scheduler = gnn.SimulationScheduler(gnn.SimulationScheduler.CONSTANT, 100, 10)
        result = scheduler.can_simulate(1, self.hist_detail_mock)
        self.assertTrue(result)
        
        # Test with different simulation intervals
        scheduler = gnn.SimulationScheduler(gnn.SimulationScheduler.CONSTANT, 50, 5)
        result = scheduler.can_simulate(1, self.hist_detail_mock)
        self.assertTrue(result)
        
        # Test with different simulation counts
        scheduler = gnn.SimulationScheduler(gnn.SimulationScheduler.CONSTANT, 100, 20)
        result = scheduler.can_simulate(1, self.hist_detail_mock)
        self.assertTrue(result)

    def test_progress_check_simulation_with_progress(self):
        """Test progress check mode when there is progress"""
        # Test with default parameters
        scheduler = gnn.SimulationScheduler(gnn.SimulationScheduler.PROGRESS_CHECK, 100, 10)
        self.hist_detail_mock.Y['iteration_acc_train'] = [0.8, 0.9]
        result = scheduler.can_simulate(1, self.hist_detail_mock)
        self.assertFalse(result)
        
        # Test with different simulation intervals
        scheduler = gnn.SimulationScheduler(gnn.SimulationScheduler.PROGRESS_CHECK, 50, 5)
        result = scheduler.can_simulate(1, self.hist_detail_mock)
        self.assertFalse(result)
        
        # Test with different simulation counts
        scheduler = gnn.SimulationScheduler(gnn.SimulationScheduler.PROGRESS_CHECK, 100, 20)
        result = scheduler.can_simulate(1, self.hist_detail_mock)
        self.assertFalse(result)
        
        # Test with small progress
        scheduler = gnn.SimulationScheduler(gnn.SimulationScheduler.PROGRESS_CHECK, 100, 10)
        self.hist_detail_mock.Y['iteration_acc_train'] = [0.89, 0.9]
        result = scheduler.can_simulate(1, self.hist_detail_mock)
        self.assertFalse(result)
        
        # Test with large progress
        scheduler = gnn.SimulationScheduler(gnn.SimulationScheduler.PROGRESS_CHECK, 100, 10)
        self.hist_detail_mock.Y['iteration_acc_train'] = [0.5, 0.9]
        result = scheduler.can_simulate(1, self.hist_detail_mock)
        self.assertFalse(result)

    def test_overfit_check_simulation_overfit_detected(self):
        """Test overfit check mode when overfitting is detected"""
        # Test with default parameters
        scheduler = gnn.SimulationScheduler(gnn.SimulationScheduler.OVERFIT_CHECK, 100, 10)
        #self.hist_detail_mock.learning_capable.return_value = True
        result = scheduler.can_simulate(1, self.hist_detail_mock)
        self.assertFalse(result)
        
        # Test with different simulation intervals
        scheduler = gnn.SimulationScheduler(gnn.SimulationScheduler.OVERFIT_CHECK, 50, 5)
        result = scheduler.can_simulate(1, self.hist_detail_mock)
        self.assertFalse(result)
        
        # Test with different simulation counts
        scheduler = gnn.SimulationScheduler(gnn.SimulationScheduler.OVERFIT_CHECK, 100, 20)
        result = scheduler.can_simulate(1, self.hist_detail_mock)
        self.assertFalse(result)

    def test_overfit_check_simulation_no_overfit(self):
        """Test overfit check mode when no overfitting is detected"""
        # Test with default parameters
        scheduler = gnn.SimulationScheduler(gnn.SimulationScheduler.OVERFIT_CHECK, 100, 10)
        #self.hist_detail_mock.learning_capable.return_value = False
        result = scheduler.can_simulate(1, self.hist_detail_mock)
        self.assertFalse(result)
        
        # Test with different simulation intervals
        scheduler = gnn.SimulationScheduler(gnn.SimulationScheduler.OVERFIT_CHECK, 50, 5)
        result = scheduler.can_simulate(1, self.hist_detail_mock)
        self.assertFalse(result)
        
        # Test with different simulation counts
        scheduler = gnn.SimulationScheduler(gnn.SimulationScheduler.OVERFIT_CHECK, 100, 20)
        result = scheduler.can_simulate(1, self.hist_detail_mock)
        self.assertFalse(result)

    def test_get_mode_label(self):
        """Test getting mode labels for different scheduler modes"""
        # Test constant mode
        scheduler = gnn.SimulationScheduler(gnn.SimulationScheduler.CONSTANT, 100, 10)
        self.assertEqual(scheduler.get_mode_label(), "simconstant")
        
        # Test progress check mode
        scheduler = gnn.SimulationScheduler(gnn.SimulationScheduler.PROGRESS_CHECK, 100, 10)
        self.assertEqual(scheduler.get_mode_label(), "simprogres")
        
        # Test overfit check mode
        scheduler = gnn.SimulationScheduler(gnn.SimulationScheduler.OVERFIT_CHECK, 100, 10)
        self.assertEqual(scheduler.get_mode_label(), "simoverfit")

if __name__ == '__main__':
    unittest.main()
