import sys
sys.path.append('.')
sys.path.append('../')
import growingnn as gnn
import numpy as np
import unittest
from testSuite import mode
import unittest
import numpy as np
from growingnn.config import config
from growingnn.structure import Model, Layer, Activations, Loss
from testSuite import mode
from testDataGenerator import TestDataGenerator

class TestConfigRuntime(unittest.TestCase):
    def setUp(self):
        # Reset config to defaults before each test
        config.reset_to_defaults()
        # Set up test data parameters
        self.datadimensionality = 2
        self.datasize = 100
        self.classes = 2
        # Generate test data
        self.x_train = TestDataGenerator.generate_x_data(self.datadimensionality, self.datasize)
        self.y_train = TestDataGenerator.generate_y_data(self.datasize, self.classes)
        self.x_test = TestDataGenerator.generate_x_data(self.datadimensionality, int(self.datasize / 2))
        self.y_test = TestDataGenerator.generate_y_data(int(self.datasize / 2), self.classes)

    def test_save_plots_change(self):
        """Test if changing save_plots affects history behavior"""
        from growingnn.structure import History
        
        # Create history object
        history = History(['accuracy', 'loss'])
        
        # Add some data
        history.append('accuracy', 0.5)
        history.append('loss', 0.1)
        
        # Test with SAVE_PLOTS=True
        config.update(SAVE_PLOTS=True)
        # This should not raise any error
        history.draw_hist('test', '.')
        
        # Test with SAVE_PLOTS=False
        config.update(SAVE_PLOTS=False)
        # This should return immediately without drawing
        history.draw_hist('test', '.')
        
    def test_progress_print_frequency_change(self):
        """Test if changing progress print frequency affects training output"""
        # Create a simple model
        model = Model(input_size=self.datadimensionality, 
                     hidden_size=3, 
                     output_size=self.classes)
        
        # Create a learning rate scheduler
        lr_scheduler = gnn.LearningRateScheduler(gnn.LearningRateScheduler.CONSTANT, 0.01)
        
        # Test with different frequencies
        config.update(PROGRESS_PRINT_FREQUENCY=1)
        # This should print every 2 iterations
        model.gradient_descent(self.x_train, self.y_train, 
                             iterations=5, 
                             lr_scheduler=lr_scheduler, 
                             quiet=False)
        
        print("Changing progress print frequency to 5")
        config.update(PROGRESS_PRINT_FREQUENCY=5)
        # This should print every 5 iterations
        model.gradient_descent(self.x_train, self.y_train, 
                             iterations=5, 
                             lr_scheduler=lr_scheduler, 
                             quiet=False)

if __name__ == '__main__':
    unittest.main() 