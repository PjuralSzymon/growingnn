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

    def test_train_save_load(self):
        x_train = np.random.randint(0, 256, (20, 32, 32, 3), dtype=np.uint8)
        y_train = np.random.randint(0, 10, (20,), dtype=np.uint8)
        labels = {'frog': 0, 'cat': 1, 'automobile': 2, 'dog': 3, 'truck': 4, 'deer': 5, 'bird': 6, 'ship': 7, 'airplane': 8, 'horse': 9}
        times = []
        for i in range(0, 3):
            start_time = time.time()
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
            generations = 3,
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
            end_time = time.time()
            time_stemp = end_time - start_time
            times.append(time_stemp)
        mean_time = np.mean(times)
        print("mean_time: ", mean_time)
        self.assertLessEqual(mean_time, 30)

if __name__ == '__main__':
    unittest.main()
