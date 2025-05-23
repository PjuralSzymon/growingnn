import sys
sys.path.append('.')
sys.path.append('../')
import growingnn as gnn
import unittest
import numpy as np
import os
import tempfile
from testSuite import mode
from testDataGenerator import TestDataGenerator

class TestTrainingScenarios(unittest.TestCase):
    def setUp(self):
        global mode
        if mode == 'cpu':
            gnn.switch_to_cpu()
        elif mode == 'gpu':
            gnn.switch_to_gpu()
            
        # Create small synthetic datasets for testing
        self.datasize = 20
        self.datadimensionality = 10
        self.classes = 3
        self.x_train = TestDataGenerator.generate_x_data(self.datadimensionality, self.datasize)
        # Use TestDataGenerator to ensure all classes are represented
        self.y_train = TestDataGenerator.generate_y_data(self.datasize, self.classes)
        self.x_test = TestDataGenerator.generate_x_data(self.datadimensionality, int(self.datasize / 2))
        self.y_test = TestDataGenerator.generate_y_data(int(self.datasize / 2), self.classes)
        
        # Create convolutional data
        self.x_conv_train = TestDataGenerator.generate_conv_x_data(self.datasize, self.datadimensionality)
        self.y_conv_train = TestDataGenerator.generate_y_data(self.datasize, self.classes)
        self.x_conv_test = TestDataGenerator.generate_conv_x_data(int(self.datasize / 2), self.datadimensionality)
        self.y_conv_test = TestDataGenerator.generate_y_data(int(self.datasize / 2), self.classes)
        
        self.labels = range(0, self.classes)
        self.optimizer = gnn.SGDOptimizer()
        self.simulation_time = 2
        self.simulation_alg = gnn.montecarlo_alg
        
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_train_dense_CPU(self):
        """Test training dense model on CPU"""
        gnn.switch_to_cpu()
        try:
            model = self.train_dense()
            self.assertIsNotNone(model)
        except Exception as e:
            self.fail(f"Dense model training on CPU failed with exception: {e}")
            
    def test_train_conv_CPU(self):
        """Test training convolutional model on CPU"""
        gnn.switch_to_cpu()
        try:
            model = self.train_conv()
            self.assertIsNotNone(model)
        except Exception as e:
            self.fail(f"Convolutional model training on CPU failed with exception: {e}")
            
    def test_train_dense_CPU_SGD(self):
        """Test training dense model with SGD optimizer"""
        gnn.switch_to_cpu()
        self.optimizer = gnn.SGDOptimizer()
        try:
            model = self.train_dense()
            self.assertIsNotNone(model)
        except Exception as e:
            self.fail(f"Dense model training with SGD failed with exception: {e}")
            
    def test_train_conv_CPU_Adam(self):
        """Test training convolutional model with Adam optimizer"""
        gnn.switch_to_cpu()
        self.optimizer = gnn.AdamOptimizer()
        try:
            model = self.train_conv()
            self.assertIsNotNone(model)
        except Exception as e:
            self.fail(f"Convolutional model training with Adam failed with exception: {e}")
            
    def test_train_conv_CPU_Adam_vs_SGD(self):
        """Test performance comparison between Adam and SGD optimizers"""
        # Create deterministic data
        np.random.seed(42)  # Set random seed for reproducibility
        x = np.random.rand(20, 20, 20, 1)  # Fixed size input
        y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1])  # Fixed class distribution
        
        # Create models with same architecture
        model_sgd = gnn.structure.Model(20, 20, 3, gnn.structure.Loss.multiclass_cross_entropy, gnn.structure.Activations.Sigmoid, 1)
        model_adam = gnn.structure.Model(20, 20, 3, gnn.structure.Loss.multiclass_cross_entropy, gnn.structure.Activations.Sigmoid, 1)
        
        # Set convolution mode
        model_sgd.set_convolution_mode((20, 20, 1), 20, 1)
        model_adam.set_convolution_mode((20, 20, 1), 20, 1)
        
        # Set optimizers
        model_sgd.optimizer = gnn.optimizers.SGDOptimizer()
        model_adam.optimizer = gnn.optimizers.AdamOptimizer()
        
        # Use same learning rate scheduler for both
        lr_scheduler = gnn.structure.LearningRateScheduler(gnn.structure.LearningRateScheduler.CONSTANT, 0.01)
        
        # Train both models
        acc_sgd, _ = model_sgd.gradient_descent(x, y, 50, lr_scheduler)
        acc_adam, _ = model_adam.gradient_descent(x, y, 50, lr_scheduler)
        
        # Adam should perform at least 80% as well as SGD
        self.assertGreaterEqual(acc_adam, acc_sgd * 0.8,
                              f"Adam optimizer should have better result than SGD: {acc_adam} > {acc_sgd * 0.8}")
        
    def test_train_dense_CPU_SGD_SIMULATION(self):
        """Test training dense model with SGD and longer simulation time"""
        gnn.switch_to_cpu()
        self.optimizer = gnn.SGDOptimizer()
        self.simulation_time = 30
        try:
            model = self.train_dense()
            self.assertIsNotNone(model)
        except Exception as e:
            self.fail(f"Dense model training with SGD and longer simulation failed with exception: {e}")
            
    def test_train_conv_CPU_Adam_SIMULATION(self):
        """Test training convolutional model with Adam and longer simulation time"""
        gnn.switch_to_cpu()
        self.optimizer = gnn.AdamOptimizer()
        self.simulation_time = 30
        try:
            model = self.train_conv()
            self.assertIsNotNone(model)
        except Exception as e:
            self.fail(f"Convolutional model training with Adam and longer simulation failed with exception: {e}")
            
    def test_train_dense_CPU_SGD_monte_carlo(self):
        """Test training dense model with SGD and Monte Carlo simulation algorithm"""
        gnn.switch_to_cpu()
        self.optimizer = gnn.SGDOptimizer()
        self.simulation_time = 10
        self.simulation_alg = gnn.montecarlo_alg
        try:
            model = self.train_dense()
            self.assertIsNotNone(model)
        except Exception as e:
            self.fail(f"Dense model training with SGD and Monte Carlo failed with exception: {e}")
            
    def test_train_dense_CPU_SGD_greedy(self):
        """Test training dense model with SGD and greedy simulation algorithm"""
        gnn.switch_to_cpu()
        self.optimizer = gnn.SGDOptimizer()
        self.simulation_time = 10
        self.simulation_alg = gnn.greedy_alg
        try:
            model = self.train_dense()
            self.assertIsNotNone(model)
        except Exception as e:
            self.fail(f"Dense model training with SGD and greedy failed with exception: {e}")
            
    def test_train_dense_CPU_SGD_random(self):
        """Test training dense model with SGD and random simulation algorithm"""
        gnn.switch_to_cpu()
        self.optimizer = gnn.SGDOptimizer()
        self.simulation_time = 10
        self.simulation_alg = gnn.random_alg
        try:
            model = self.train_dense()
            self.assertIsNotNone(model)
        except Exception as e:
            self.fail(f"Dense model training with SGD and random failed with exception: {e}")
            
    def test_train_with_different_epochs(self):
        """Test training with different numbers of epochs"""
        epochs_list = [1, 2, 5, 10]
        
        for epochs in epochs_list:
            try:
                model = gnn.trainer.train(
                    x_train=self.x_train,
                    y_train=self.y_train,
                    x_test=self.x_test,
                    y_test=self.y_test,
                    labels=self.labels,
                    input_paths=1,
                    path=self.temp_dir,
                    model_name=f"test_model_epochs_{epochs}",
                    epochs=epochs,
                    generations=1,
                    input_size=self.datadimensionality,
                    hidden_size=self.datadimensionality,
                    output_size=self.classes,
                    input_shape=None,
                    kernel_size=None,
                    batch_size=1,
                    simulation_scheduler=gnn.SimulationScheduler(
                        gnn.SimulationScheduler.CONSTANT, 
                        simulation_time=1, 
                        simulation_epochs=1
                    ),
                    deepth=None,
                    simulation_alg=self.simulation_alg,
                    optimizer=self.optimizer
                )
                self.assertIsNotNone(model)
            except Exception as e:
                self.fail(f"Training with {epochs} epochs failed with exception: {e}")
                
    def test_train_with_different_generations(self):
        """Test training with different numbers of generations"""
        generations_list = [1, 2, 5]
        
        for generations in generations_list:
            try:
                model = gnn.trainer.train(
                    x_train=self.x_train,
                    y_train=self.y_train,
                    x_test=self.x_test,
                    y_test=self.y_test,
                    labels=self.labels,
                    input_paths=1,
                    path=self.temp_dir,
                    model_name=f"test_model_generations_{generations}",
                    epochs=2,
                    generations=generations,
                    input_size=self.datadimensionality,
                    hidden_size=self.datadimensionality,
                    output_size=self.classes,
                    input_shape=None,
                    kernel_size=None,
                    batch_size=1,
                    simulation_scheduler=gnn.SimulationScheduler(
                        gnn.SimulationScheduler.CONSTANT, 
                        simulation_time=1, 
                        simulation_epochs=1
                    ),
                    deepth=None,
                    simulation_alg=self.simulation_alg,
                    optimizer=self.optimizer
                )
                self.assertIsNotNone(model)
            except Exception as e:
                self.fail(f"Training with {generations} generations failed with exception: {e}")
                
    def test_train_with_different_network_depths(self):
        """Test training with different network depths"""
        depths = [None, 1, 2, 3]
        
        for depth in depths:
            try:
                model = gnn.trainer.train(
                    x_train=self.x_train,
                    y_train=self.y_train,
                    x_test=self.x_test,
                    y_test=self.y_test,
                    labels=self.labels,
                    input_paths=1,
                    path=self.temp_dir,
                    model_name=f"test_model_depth_{depth}",
                    epochs=2,
                    generations=1,
                    input_size=self.datadimensionality,
                    hidden_size=self.datadimensionality,
                    output_size=self.classes,
                    input_shape=None,
                    kernel_size=None,
                    batch_size=1,
                    simulation_scheduler=gnn.SimulationScheduler(
                        gnn.SimulationScheduler.CONSTANT, 
                        simulation_time=1, 
                        simulation_epochs=1
                    ),
                    deepth=depth,
                    simulation_alg=self.simulation_alg,
                    optimizer=self.optimizer
                )
                self.assertIsNotNone(model)
            except Exception as e:
                self.fail(f"Training with depth {depth} failed with exception: {e}")
                
    def test_train_with_different_kernel_sizes(self):
        """Test training with different kernel sizes for convolutional layers"""
        kernel_sizes = [2, 3, 4]
        
        for kernel_size in kernel_sizes:
            try:
                model = gnn.trainer.train(
                    x_train=self.x_conv_train,
                    y_train=self.y_conv_train,
                    x_test=self.x_conv_test,
                    y_test=self.y_conv_test,
                    labels=self.labels,
                    input_paths=1,
                    path=self.temp_dir,
                    model_name=f"test_model_kernel_{kernel_size}",
                    epochs=2,
                    generations=1,
                    input_size=self.datadimensionality,
                    hidden_size=self.datadimensionality,
                    output_size=self.classes,
                    input_shape=(self.datadimensionality, self.datadimensionality, 1),
                    kernel_size=kernel_size,
                    batch_size=1,
                    simulation_scheduler=gnn.SimulationScheduler(
                        gnn.SimulationScheduler.CONSTANT, 
                        simulation_time=1, 
                        simulation_epochs=1
                    ),
                    deepth=1,
                    simulation_alg=self.simulation_alg,
                    optimizer=self.optimizer
                )
                self.assertIsNotNone(model)
            except Exception as e:
                self.fail(f"Training with kernel size {kernel_size} failed with exception: {e}")
                
    def train_conv(self):
        """Helper method to train a convolutional model"""
        return gnn.trainer.train(
            x_train=self.x_conv_train,
            y_train=self.y_conv_train,
            x_test=self.x_conv_test,
            y_test=self.y_conv_test,
            labels=self.labels,
            input_paths=1,
            path=self.temp_dir,
            model_name="GNN_model",
            epochs=3,
            generations=3,
            input_size=self.datadimensionality,
            hidden_size=self.datadimensionality,
            output_size=self.classes,
            input_shape=(self.datadimensionality, self.datadimensionality, 1),
            kernel_size=2,
            batch_size=1,
            simulation_scheduler=gnn.SimulationScheduler(
                gnn.SimulationScheduler.PROGRESS_CHECK, 
                simulation_time=self.simulation_time, 
                simulation_epochs=2
            ), 
            deepth=2,
            simulation_alg=self.simulation_alg,
            optimizer=self.optimizer
        )

    def train_dense(self):
        """Helper method to train a dense model"""
        return gnn.trainer.train(
            x_train=self.x_train,
            y_train=self.y_train,
            x_test=self.x_test,
            y_test=self.y_test,
            labels=self.labels,
            input_paths=1,
            path=self.temp_dir,
            model_name="GNN_model",
            epochs=3,
            generations=3,
            input_size=self.datadimensionality,
            hidden_size=self.datadimensionality,
            output_size=self.classes,
            input_shape=None,
            kernel_size=None,
            batch_size=1,
            simulation_scheduler=gnn.SimulationScheduler(
                gnn.SimulationScheduler.PROGRESS_CHECK, 
                simulation_time=self.simulation_time, 
                simulation_epochs=2
            ), 
            deepth=None,
            simulation_alg=self.simulation_alg,
            optimizer=self.optimizer
        )


if __name__ == '__main__':
    unittest.main() 