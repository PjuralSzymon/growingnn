import sys
sys.path.append('.')
sys.path.append('../')
import growingnn as gnn
import unittest
import numpy as np
import os
import tempfile
from testSuite import mode

class TestTrainer(unittest.TestCase):
    def setUp(self):
        global mode
        if mode == 'cpu':
            gnn.switch_to_cpu()
        elif mode == 'gpu':
            gnn.switch_to_gpu()
            
        # Create small synthetic datasets for testing
        self.datasize = 20
        self.datadimensionality = 5
        self.classes = 3
        self.x_train = np.random.random((self.datadimensionality, self.datasize))
        self.y_train = np.random.randint(self.classes, size=(self.datasize,))
        self.x_test = np.random.random((self.datadimensionality, int(self.datasize / 2)))
        self.y_test = np.random.randint(self.classes, size=(int(self.datasize / 2),))
        
        # Create convolutional data
        self.x_conv_train = np.random.random((self.datasize, self.datadimensionality, self.datadimensionality, 1))
        self.y_conv_train = np.random.randint(self.classes, size=(self.datasize,))
        self.x_conv_test = np.random.random((int(self.datasize / 2), self.datadimensionality, self.datadimensionality, 1))
        self.y_conv_test = np.random.randint(self.classes, size=(int(self.datasize / 2),))
        
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
        
    def test_train_basic(self):
        """Test basic training functionality with minimal parameters"""
        try:
            model = gnn.trainer.train(
                x_train=self.x_train,
                y_train=self.y_train,
                x_test=self.x_test,
                y_test=self.y_test,
                labels=self.labels,
                input_paths=1,
                path=self.temp_dir,
                model_name="test_model",
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
                deepth=None,
                simulation_alg=self.simulation_alg,
                optimizer=self.optimizer
            )
            self.assertIsNotNone(model)
        except Exception as e:
            self.fail(f"Basic training failed with exception: {e}")
            
    def test_train_with_convolution(self):
        """Test training with convolutional layers"""
        try:
            model = gnn.trainer.train(
                x_train=self.x_conv_train,
                y_train=self.y_conv_train,
                x_test=self.x_conv_test,
                y_test=self.y_conv_test,
                labels=self.labels,
                input_paths=1,
                path=self.temp_dir,
                model_name="test_conv_model",
                epochs=2,
                generations=1,
                input_size=self.datadimensionality,
                hidden_size=self.datadimensionality,
                output_size=self.classes,
                input_shape=(self.datadimensionality, self.datadimensionality, 1),
                kernel_size=2,
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
            self.fail(f"Convolutional training failed with exception: {e}")
            
    def test_train_with_different_optimizers(self):
        """Test training with different optimizers"""
        optimizers = [
            gnn.SGDOptimizer(),
            gnn.AdamOptimizer()
        ]
        
        for optimizer in optimizers:
            try:
                model = gnn.trainer.train(
                    x_train=self.x_train,
                    y_train=self.y_train,
                    x_test=self.x_test,
                    y_test=self.y_test,
                    labels=self.labels,
                    input_paths=1,
                    path=self.temp_dir,
                    model_name=f"test_model_{optimizer.__class__.__name__}",
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
                    deepth=None,
                    simulation_alg=self.simulation_alg,
                    optimizer=optimizer
                )
                self.assertIsNotNone(model)
            except Exception as e:
                self.fail(f"Training with {optimizer.__class__.__name__} failed with exception: {e}")
                
    def test_train_with_different_simulation_algorithms(self):
        """Test training with different simulation algorithms"""
        simulation_algs = [
            gnn.montecarlo_alg,
            gnn.greedy_alg,
            gnn.random_alg
        ]
        
        for alg in simulation_algs:
            try:
                model = gnn.trainer.train(
                    x_train=self.x_train,
                    y_train=self.y_train,
                    x_test=self.x_test,
                    y_test=self.y_test,
                    labels=self.labels,
                    input_paths=1,
                    path=self.temp_dir,
                    model_name=f"test_model_{alg.__name__}",
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
                    deepth=None,
                    simulation_alg=alg,
                    optimizer=self.optimizer
                )
                self.assertIsNotNone(model)
            except Exception as e:
                self.fail(f"Training with {alg.__name__} failed with exception: {e}")
                
    def test_train_with_different_simulation_schedulers(self):
        """Test training with different simulation schedulers"""
        scheduler_modes = [
            gnn.SimulationScheduler.CONSTANT,
            gnn.SimulationScheduler.PROGRESS_CHECK,
            gnn.SimulationScheduler.OVERFIT_CHECK
        ]
        
        for mode in scheduler_modes:
            try:
                model = gnn.trainer.train(
                    x_train=self.x_train,
                    y_train=self.y_train,
                    x_test=self.x_test,
                    y_test=self.y_test,
                    labels=self.labels,
                    input_paths=1,
                    path=self.temp_dir,
                    model_name=f"test_model_scheduler_{mode}",
                    epochs=2,
                    generations=1,
                    input_size=self.datadimensionality,
                    hidden_size=self.datadimensionality,
                    output_size=self.classes,
                    input_shape=None,
                    kernel_size=None,
                    batch_size=1,
                    simulation_scheduler=gnn.SimulationScheduler(
                        mode, 
                        simulation_time=1, 
                        simulation_epochs=1
                    ),
                    deepth=None,
                    simulation_alg=self.simulation_alg,
                    optimizer=self.optimizer
                )
                self.assertIsNotNone(model)
            except Exception as e:
                self.fail(f"Training with scheduler mode {mode} failed with exception: {e}")
                
    def test_train_with_different_learning_rate_schedulers(self):
        """Test training with different learning rate schedulers"""
        lr_schedulers = [
            gnn.LearningRateScheduler(gnn.LearningRateScheduler.CONSTANT, 0.01),
            gnn.LearningRateScheduler(gnn.LearningRateScheduler.PROGRESIVE, 0.01, 0.2),
            gnn.LearningRateScheduler(gnn.LearningRateScheduler.PROGRESIVE_PARABOIDAL, 0.01, 0.2)
        ]
        
        for scheduler in lr_schedulers:
            try:
                model = gnn.trainer.train(
                    x_train=self.x_train,
                    y_train=self.y_train,
                    x_test=self.x_test,
                    y_test=self.y_test,
                    labels=self.labels,
                    input_paths=1,
                    path=self.temp_dir,
                    model_name=f"test_model_lr_{scheduler.mode}",
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
                    deepth=None,
                    simulation_alg=self.simulation_alg,
                    optimizer=self.optimizer,
                    lr_scheduler=scheduler
                )
                self.assertIsNotNone(model)
            except Exception as e:
                self.fail(f"Training with learning rate scheduler {scheduler.mode} failed with exception: {e}")
                
    def test_train_with_different_activation_functions(self):
        """Test training with different activation functions"""
        activation_funs = [
            gnn.Activations.ReLu,
            gnn.Activations.leaky_ReLu,
            gnn.Activations.Sigmoid,
            gnn.Activations.SoftMax
        ]
        
        for act_fun in activation_funs:
            try:
                model = gnn.trainer.train(
                    x_train=self.x_train,
                    y_train=self.y_train,
                    x_test=self.x_test,
                    y_test=self.y_test,
                    labels=self.labels,
                    input_paths=1,
                    path=self.temp_dir,
                    model_name=f"test_model_act_{act_fun.__name__}",
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
                    deepth=None,
                    simulation_alg=self.simulation_alg,
                    optimizer=self.optimizer,
                    activation_fun=act_fun
                )
                self.assertIsNotNone(model)
            except Exception as e:
                self.fail(f"Training with activation function {act_fun.__name__} failed with exception: {e}")
                
    def test_train_with_different_loss_functions(self):
        """Test training with different loss functions"""
        loss_functions = [
            gnn.Loss.MSE,
            gnn.Loss.multiclass_cross_entropy
        ]
        
        for loss_fun in loss_functions:
            try:
                model = gnn.trainer.train(
                    x_train=self.x_train,
                    y_train=self.y_train,
                    x_test=self.x_test,
                    y_test=self.y_test,
                    labels=self.labels,
                    input_paths=1,
                    path=self.temp_dir,
                    model_name=f"test_model_loss_{loss_fun.__name__}",
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
                    deepth=None,
                    simulation_alg=self.simulation_alg,
                    optimizer=self.optimizer,
                    loss_function=loss_fun
                )
                self.assertIsNotNone(model)
            except Exception as e:
                self.fail(f"Training with loss function {loss_fun.__name__} failed with exception: {e}")
                
    def test_train_with_multiple_input_paths(self):
        """Test training with multiple input paths"""
        try:
            # Create multiple input paths
            x_train_multi = [self.x_train, self.x_train]
            x_test_multi = [self.x_test, self.x_test]
            
            model = gnn.trainer.train(
                x_train=x_train_multi,
                y_train=self.y_train,
                x_test=x_test_multi,
                y_test=self.y_test,
                labels=self.labels,
                input_paths=2,
                path=self.temp_dir,
                model_name="test_model_multi_input",
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
                deepth=None,
                simulation_alg=self.simulation_alg,
                optimizer=self.optimizer
            )
            self.assertIsNotNone(model)
        except Exception as e:
            self.fail(f"Training with multiple input paths failed with exception: {e}")
            
    def test_train_with_different_batch_sizes(self):
        """Test training with different batch sizes"""
        batch_sizes = [1, 2, 5, 10]
        
        for batch_size in batch_sizes:
            try:
                model = gnn.trainer.train(
                    x_train=self.x_train,
                    y_train=self.y_train,
                    x_test=self.x_test,
                    y_test=self.y_test,
                    labels=self.labels,
                    input_paths=1,
                    path=self.temp_dir,
                    model_name=f"test_model_batch_{batch_size}",
                    epochs=2,
                    generations=1,
                    input_size=self.datadimensionality,
                    hidden_size=self.datadimensionality,
                    output_size=self.classes,
                    input_shape=None,
                    kernel_size=None,
                    batch_size=batch_size,
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
                self.fail(f"Training with batch size {batch_size} failed with exception: {e}")


if __name__ == '__main__':
    unittest.main() 