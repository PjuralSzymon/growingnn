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
        print("self.x_train: ", self.x_train.shape)
        print("self.y_train: ", self.y_train.shape)

    def test_train_dense_GPU(self):
        mode = getattr(self, 'mode', 'cpu')  # Default to 'cpu' if 'mode' is not set
        if mode == 'cpu':
            return
        # Wykonywanie treningu modelu z małym zbiorem danych
        gnn.switch_to_gpu()
        try:
            self.train_dense()
        except Exception as e:
            self.fail(f"Model training failed with exception: {e}")

    def test_train_dense_CPU(self):
        # Wykonywanie treningu modelu z małym zbiorem danych
        gnn.switch_to_cpu()
        print(gnn.IS_CUPY)
        try:
            self.train_dense()
        except Exception as e:
            self.fail(f"Model training failed with exception: {e}")

    def test_train_conv_GPU(self):
        mode = getattr(self, 'mode', 'cpu')  # Default to 'cpu' if 'mode' is not set
        if mode == 'cpu':
            return
        # Wykonywanie treningu modelu z małym zbiorem danych
        gnn.switch_to_gpu()
        try:
            self.train_conv()
        except Exception as e:
            self.fail(f"Model training failed with exception: {e}")

    def test_train_conv_CPU(self):
        # Wykonywanie treningu modelu z małym zbiorem danych
        gnn.switch_to_cpu()
        print(gnn.IS_CUPY)
        try:
            self.train_conv()
        except Exception as e:
            self.fail(f"Model training failed with exception: {e}")

    def train_conv(self):
        gnn.trainer.train(
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
            simulation_scheduler = SimulationScheduler(SimulationScheduler.PROGRESS_CHECK, simulation_time = 2, simulation_epochs = 2), 
            deepth=2
        )

    def train_dense(self):
        gnn.trainer.train(
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
            simulation_scheduler = SimulationScheduler(SimulationScheduler.PROGRESS_CHECK, simulation_time = 2, simulation_epochs = 2), 
            deepth=None
        )


if __name__ == '__main__':
    unittest.main()
