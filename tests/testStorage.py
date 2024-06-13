import sys
sys.path.append('.')
sys.path.append('../')
import growingnn as gnn
import numpy as np
import time
import unittest
import pandas as pd

shape = 5000

class TestingStorage(unittest.TestCase):

    def setUp(self):
        mode = getattr(self, 'mode', 'cpu')  # Default to 'cpu' if 'mode' is not set
        if mode == 'cpu':
            gnn.switch_to_cpu()
        elif mode == 'gpu':
            gnn.switch_to_gpu()
            
    def test_base_save_load(self):
        M = gnn.structure.Model(3, 3, 1, gnn.structure.Loss.multiclass_cross_entropy, gnn.structure.Activations.Sigmoid, 1)
        x = np.random.rand(3, 3)
        y = np.random.randint(2, size=(50,))
        output1 = M.forward_prop(x)
        gnn.Storage.saveModel(M, "here.json")
        M_loaded = gnn.Storage.loadModel("here.json")
        output2 = M_loaded.forward_prop(x)
        self.assertAlmostEqual(np.sum(output1 - output2), 0)

    def test_train_save_load(self):
        x_train = np.random.randint(0, 256, (90, 32, 32, 3), dtype=np.uint8)
        y_train = np.random.randint(0, 10, (90,), dtype=np.uint8)
        labels = {'frog': 0, 'cat': 1, 'automobile': 2, 'dog': 3, 'truck': 4, 'deer': 5, 'bird': 6, 'ship': 7, 'airplane': 8, 'horse': 9}
        for g in range(1, 10, 4):
            M = gnn.trainer.train(
            x_train = x_train, 
            y_train = y_train, 
            x_test= x_train,
            y_test= y_train,
            labels=labels,
            input_paths = 1,
            path = "work/results/cifar10gpu_paraboidals/", 
            model_name = "GNN_model", #"GNN_model", 
            epochs = 2, 
            generations = g,
            simulation_scheduler = gnn.SimulationScheduler(gnn.SimulationScheduler.PROGRESS_CHECK, simulation_time = 6, simulation_epochs = 2),
            simulation_alg = gnn.montecarlo_alg,
            sim_set_generator = gnn.simulation.create_simulation_set_SAMLE,
            lr_scheduler = gnn.LearningRateScheduler(gnn.LearningRateScheduler.PROGRESIVE_PARABOIDAL, 0.05, 0.8),
            input_size = 32 * 32, 
            hidden_size = 300, 
            output_size = 10, 
            input_shape = (32, 32, 3), 
            kernel_size = 3, 
            deepth = 2, 
            batch_size = 5 * 128, 
            simulation_set_size = 5,
            simulation_score = gnn.Simulation_score(gnn.Simulation_score.ACCURACY))
            output1 = M.forward_prop(x_train[:1])
            gnn.Storage.saveModel(M, "tmp.json")
            M_loaded = gnn.Storage.loadModel("tmp.json")
            output2 = M_loaded.forward_prop(x_train[:1])
            self.assertAlmostEqual(np.sum(output1 - output2), 0)
            

if __name__ == '__main__':
    unittest.main()
