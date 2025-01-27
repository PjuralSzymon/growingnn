import sys
sys.path.append('.')
sys.path.append('../')
import unittest
import numpy as np
import growingnn as gnn
from growingnn.structure import SimulationScheduler
from testSuite import mode

class TestModelTraining(unittest.TestCase):

    def setUp(self):
        global mode
        if mode == 'cpu':
            gnn.switch_to_cpu()
        elif mode == 'gpu':
            gnn.switch_to_gpu()

        self.datasize = 20
        self.datadimensionality = 10
        self.classes = 3
        self.x_train = np.random.random((self.datadimensionality, self.datasize))
        self.y_train = np.random.randint(self.classes, size=(self.datasize, ))
        self.x_test = np.random.random((self.datadimensionality, int(self.datasize / 2)))
        self.y_test = np.random.randint(self.classes, size=(int(self.datasize / 2), ))

        self.x_conv_train = np.random.random((self.datasize, self.datadimensionality, self.datadimensionality, 1))
        self.y_conv_train = np.random.randint(self.classes, size=(self.datasize, ))
        self.x_conv_test = np.random.random((int(self.datasize / 2), self.datadimensionality, self.datadimensionality, 1))
        self.y_conv_test = np.random.randint(self.classes, size=(int(self.datasize / 2), ))
        self.labels = range(0, self.classes)
        self.optimizer = gnn.SGDOptimizer()
        self.simulation_time = 2
        self.simulation_alg = gnn.montecarlo_alg
        print("self.x_train: ", self.x_train.shape)
        print("self.y_train: ", self.y_train.shape)

    # def test_train_dense_GPU(self):
    #     mode = getattr(self, 'mode', 'cpu')  # Default to 'cpu' if 'mode' is not set
    #     if mode == 'cpu':
    #         return
    #     # Wykonywanie treningu modelu z małym zbiorem danych
    #     gnn.switch_to_gpu()
    #     try:
    #         self.train_dense()
    #     except Exception as e:
    #         self.fail(f"Model training failed with exception: {e}")

    def test_train_dense_CPU(self):
        # Wykonywanie treningu modelu z małym zbiorem danych
        gnn.switch_to_cpu()
        self.train_dense()
        try:
            self.train_dense()
        except Exception as e:
            self.fail(f"Model training failed with exception: {e}")

    # def test_train_conv_GPU(self):
    #     mode = getattr(self, 'mode', 'cpu')  # Default to 'cpu' if 'mode' is not set
    #     if mode == 'cpu':
    #         return
    #     # Wykonywanie treningu modelu z małym zbiorem danych
    #     gnn.switch_to_gpu()
    #     try:
    #         self.train_conv()
    #     except Exception as e:
    #         self.fail(f"Model training failed with exception: {e}")

    def test_train_conv_CPU(self):
        # Wykonywanie treningu modelu z małym zbiorem danych
        gnn.switch_to_cpu()
        self.train_conv()
        try:
            self.train_conv()
        except Exception as e:
            self.fail(f"Model training failed with exception: {e}")

    def test_train_dense_CPU_SGD(self):
        # Wykonywanie treningu modelu z małym zbiorem danych
        gnn.switch_to_cpu()
        self.optimizer = gnn.SGDOptimizer()
        self.train_dense()
        try:
            self.train_dense()
        except Exception as e:
            self.fail(f"Model training failed with exception: {e}")

    def test_train_conv_CPU_Adam(self):
        # Wykonywanie treningu modelu z małym zbiorem danych
        gnn.switch_to_cpu()
        self.optimizer = gnn.AdamOptimizer()
        self.train_conv()
        try:
            self.train_conv()
        except Exception as e:
            self.fail(f"Model training failed with exception: {e}") #acc = Model.get_accuracy(Model.get_predictions(self.forward_prop(X)),Y)

    def test_train_conv_CPU_Adam_vs_SGD(self):
        # Wykonywanie treningu modelu z małym zbiorem danych
        gnn.switch_to_cpu()
        self.optimizer = gnn.AdamOptimizer()
        model_adam = self.train_dense()

        self.optimizer = gnn.SGDOptimizer()
        model_sgd = self.train_dense()

        acc_adam = gnn.Model.get_accuracy(gnn.Model.get_predictions(model_adam.forward_prop(self.x_train)), self.y_train)
        acc_sgd = gnn.Model.get_accuracy(gnn.Model.get_predictions(model_sgd.forward_prop(self.x_train)), self.y_train)
        
        self.assertEqual(acc_adam >= acc_sgd * 0.9, True, "Adam optimzier should have better result than SGD" + str(acc_adam) + " > " + str(acc_sgd))
        try:
            self.train_conv()
        except Exception as e:
            self.fail(f"Model training failed with exception: {e}") #acc = Model.get_accuracy(Model.get_predictions(self.forward_prop(X)),Y)

    def test_train_dense_CPU_SGD_SIMULATION(self):
        # Wykonywanie treningu modelu z małym zbiorem danych
        gnn.switch_to_cpu()
        self.optimizer = gnn.SGDOptimizer()
        self.simulation_time = 30
        self.train_dense()
        try:
            self.train_dense()
        except Exception as e:
            self.fail(f"Model training failed with exception: {e}")

    def test_train_conv_CPU_Adam_SIMULATION(self):
        # Wykonywanie treningu modelu z małym zbiorem danych
        gnn.switch_to_cpu()
        self.optimizer = gnn.AdamOptimizer()
        self.simulation_time = 30
        self.train_conv()
        try:
            self.train_conv()
        except Exception as e:
            self.fail(f"Model training failed with exception: {e}") #acc = Model.get_accuracy(Model.get_predictions(self.forward_prop(X)),Y)

    def test_train_dense_CPU_SGD_monte_carlo(self):
        # Wykonywanie treningu modelu z małym zbiorem danych
        gnn.switch_to_cpu()
        self.optimizer = gnn.SGDOptimizer()
        self.simulation_time = 10
        self.simulation_alg = gnn.montecarlo_alg
        self.train_dense()
        try:
            self.train_dense()
        except Exception as e:
            self.fail(f"Model training failed with exception: {e}")

    def test_train_dense_CPU_SGD_greedy(self):
        # Wykonywanie treningu modelu z małym zbiorem danych
        gnn.switch_to_cpu()
        self.optimizer = gnn.SGDOptimizer()
        self.simulation_time = 10
        self.simulation_alg = gnn.greedy_alg
        self.train_dense()
        try:
            self.train_dense()
        except Exception as e:
            self.fail(f"Model training failed with exception: {e}")
            
    def test_train_dense_CPU_SGD_random(self):
        # Wykonywanie treningu modelu z małym zbiorem danych
        gnn.switch_to_cpu()
        self.optimizer = gnn.SGDOptimizer()
        self.simulation_time = 10
        self.simulation_alg = gnn.random_alg
        self.train_dense()
        try:
            self.train_dense()
        except Exception as e:
            self.fail(f"Model training failed with exception: {e}")
            
    def train_conv(self):
        return gnn.trainer.train(
            x_train=self.x_conv_train,
            y_train=self.y_conv_train,
            x_test=self.x_conv_test,
            y_test=self.y_conv_test,
            labels=self.labels,
            input_paths=1,
            path="./result",
            model_name="GNN_model",
            epochs=3,
            generations=3,
            input_size=self.datadimensionality,
            hidden_size=self.datadimensionality,
            output_size=self.classes,
            input_shape=(self.datadimensionality, self.datadimensionality, 1),
            kernel_size=2,
            batch_size=1,
            simulation_scheduler = SimulationScheduler(SimulationScheduler.PROGRESS_CHECK, simulation_time = self.simulation_time, simulation_epochs = 2), 
            deepth=2,
            simulation_alg=self.simulation_alg,
            optimizer=self.optimizer
        )

    def train_dense(self):
        return gnn.trainer.train(
            x_train=self.x_train,
            y_train=self.y_train,
            x_test=self.x_test,
            y_test=self.y_test,
            labels=self.labels,
            input_paths=1,
            path="./result",
            model_name="GNN_model",
            epochs=3,
            generations=3,
            input_size=self.datadimensionality,
            hidden_size=self.datadimensionality,
            output_size=self.classes,
            input_shape=None,
            kernel_size=None,
            batch_size=1,
            simulation_scheduler = SimulationScheduler(SimulationScheduler.PROGRESS_CHECK, simulation_time = self.simulation_time, simulation_epochs = 2), 
            deepth=None,
            simulation_alg=self.simulation_alg,
            optimizer=self.optimizer
        )


if __name__ == '__main__':
    unittest.main()
