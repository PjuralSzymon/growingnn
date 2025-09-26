import tempfile
import sys
sys.path.append('.')
sys.path.append('../')
import growingnn as gnn
import unittest
import numpy as np
import os
import tempfile

class TestRegressionTrainer(unittest.TestCase):
    def test_gradient_descent_regression_linear_easy(self):
        x_train = np.arange(1, 5)
        y_train = x_train * 3
        loss = self.gradient_descent_regression(x_train, y_train, 100)
        self.assertLess(loss, 5.0, f"Regression loss too high: {loss}")

    def test_gradient_descent_regression_linear_hard(self):
        x_train = np.arange(1, 5)
        y_train = x_train ** 2
        loss = self.gradient_descent_regression(x_train, y_train, 200)
        self.assertLess(loss, 5.0, f"Regression loss too high: {loss}")

    def gradient_descent_regression(self, x_train, y_train, epochs=10, lr = 0.001):
        model = gnn.Model(
            input_size=1,
            hidden_size=20,
            output_size=1,
            loss_function=gnn.Loss.MSE,
            activation_fun=gnn.Activations.Linear,
            output_activation_fun=gnn.Activations.Linear,
            input_paths=1,
            _optimizer=gnn.SGDOptimizer()
        )
        lr_scheduler = gnn.LearningRateScheduler(gnn.LearningRateScheduler.CONSTANT, lr)
        final_loss, history = model.gradient_descent(
            X=x_train,
            Y=y_train,
            iterations=epochs,
            lr_scheduler=lr_scheduler,
            quiet=True,
            one_hot_needed=False
        )
        return final_loss

    def test_trainer_easy(self):
        x_train = np.arange(1, 50)
        y_train = x_train * 3
        try:
            self.train_regression(x_train, y_train, 10, 2, 0.001)
        except Exception as e:
            self.fail(f"Model training failed with exception: {e}")

    def test_trainer_easy(self):
        x_train = np.arange(1, 50)
        y_train = x_train ** 3
        try:
            self.train_regression(x_train, y_train, 20, 3, 0.001)
        except Exception as e:
            self.fail(f"Model training failed with exception: {e}")

    def train_regression(self, x_train, y_train, epochs=10, generations=2, lr = 0.001):
        temp_dir = tempfile.mkdtemp()
        model = gnn.trainer.train(
            x_train=x_train,
            y_train=y_train,
            x_test=x_train,
            y_test=y_train,
            labels=['Y'],
            input_paths=1,
            path=temp_dir,
            model_name="test_model_multi_input",
            epochs=epochs,
            generations=generations,
            input_size=1,
            hidden_size=5,
            output_size=1,
            input_shape=None,
            kernel_size=None,
            batch_size=10,
            activation_fun=gnn.Activations.Linear,
            output_activation_fun=gnn.Activations.Linear,
            loss_function = gnn.Loss.MSE,
            lr_scheduler = gnn.LearningRateScheduler(gnn.LearningRateScheduler.CONSTANT, lr, 0.5),
            simulation_scheduler=gnn.SimulationScheduler(
                gnn.SimulationScheduler.CONSTANT, 
                simulation_time=5, 
                simulation_epochs=int(epochs/2)
            ),
            simulation_score = gnn.Simulation_score(weight_acc = 0.0, weight_loss = 1.0),
            deepth=None,
            quiet=True,
            simulation_alg=gnn.montecarlo_alg,
            optimizer=gnn.SGDOptimizer()
            )

if __name__ == '__main__':
    unittest.main() 