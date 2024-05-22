import asyncio
import sys
sys.path.append('.')
sys.path.append('../')
import growingnn as gnn
import numpy as np
import unittest
import pandas as pd
import random

shape = 10
epochs = 5

class TestingTrain(unittest.TestCase):

    def test_drawing_graphs(self):
        success = False
        try:
            M = gnn.structure.Model(shape, shape, 2, gnn.structure.Loss.multiclass_cross_entropy, gnn.structure.Activations.Sigmoid, 1)
            gnn.painter.draw(M, "testimage.html")
            success = True
        except Exception as e:
            success = False
        self.assertEqual(success, True)

    def test_base_structure(self):
        M = gnn.structure.Model(shape, shape, 2, gnn.structure.Loss.multiclass_cross_entropy, gnn.structure.Activations.Sigmoid, 1)
        x = np.asarray(np.random.rand(shape, shape))
        for _ in range(10):
            output = M.forward_prop(x * float(np.random.ranf(1)))
        self.assertEqual(output.any(), True)

    def test_res_structure(self):
        M = gnn.structure.Model(shape, shape, 2, gnn.structure.Loss.multiclass_cross_entropy, gnn.structure.Activations.Sigmoid, 1)
        M.add_res_layer('init_0', 1)
        x = np.asarray(np.random.rand(shape, shape))
        for _ in range(10):
            output = M.forward_prop(x * float(np.random.ranf(1)))
        self.assertEqual(output.any(), True)

    def test_simple_train(self):
        M = gnn.structure.Model(shape,shape,2, gnn.structure.Loss.multiclass_cross_entropy, gnn.structure.Activations.Sigmoid, 1)
        x = np.random.rand(shape, shape)
        y = np.random.randint(2, size=(shape,))
        lr_scheduler = gnn.structure.LearningRateScheduler(gnn.structure.LearningRateScheduler.PROGRESIVE, 0.03, 0.8)
        gnn.painter.draw(M, "input_test.html")
        acc, _ = M.gradient_descent(x, y, epochs, lr_scheduler)
        self.assertEqual(acc >= 0.4, True)

    def test_actions(self):
        M = gnn.structure.Model(shape,shape,2, gnn.structure.Loss.multiclass_cross_entropy, gnn.structure.Activations.Sigmoid, 1)
        x = np.random.rand(shape, shape)
        y = np.random.randint(2, size=(shape,))
        lr_scheduler = gnn.structure.LearningRateScheduler(gnn.structure.LearningRateScheduler.PROGRESIVE, 0.03, 0.8)
        for i in range(0,10):
            all_actions = gnn.action.Action.generate_all_actions(M)
            new_action = random.choice(all_actions)        
            new_action.execute(M)
            M.gradient_descent(x, y, epochs, lr_scheduler, True)
        delete_layer_actions = gnn.action.Del_Layer.generate_all_actions(M)
        while len(delete_layer_actions) > 0:
            new_action = random.choice(delete_layer_actions)        
            new_action.execute(M)
            M.gradient_descent(x, y, epochs, lr_scheduler, True)
            delete_layer_actions = gnn.action.Del_Layer.generate_all_actions(M)
        acc, _ = M.gradient_descent(x, y, epochs, lr_scheduler, True)
        print("test_actions result acc: ", acc)
        self.assertEqual(acc >= 0.01, True)

    def test_montecarlo(self):
        M = gnn.structure.Model(shape,shape,2, gnn.structure.Loss.multiclass_cross_entropy, gnn.structure.Activations.Sigmoid, 1)
        x = np.random.rand(shape, shape)
        y = np.random.randint(2, size=(shape,))
        lr_scheduler = gnn.structure.LearningRateScheduler(gnn.structure.LearningRateScheduler.PROGRESIVE, 0.03, 0.8)
        loop = asyncio.get_event_loop()
        action, deepth, rollouts = loop.run_until_complete(gnn.montecarlo_alg.get_action(M, 5, 2, x, y, gnn.Simulation_score()))
        acc, _ = M.gradient_descent(x, y, epochs, lr_scheduler, True)
        print("test_montecarlo result acc: ", acc)
        self.assertEqual(acc >= 0.01, True)

    def test_MCTS_simulation(self):
        M = gnn.structure.Model(shape,shape,2, gnn.structure.Loss.multiclass_cross_entropy, gnn.structure.Activations.Sigmoid, 1)
        x = np.random.rand(shape, shape)
        y = np.random.randint(2, size=(shape,))
        lr_scheduler = gnn.structure.LearningRateScheduler(gnn.structure.LearningRateScheduler.PROGRESIVE, 0.03, 0.8)
        acc = M.gradient_descent(x, y, epochs, lr_scheduler, True)
        for i in range(0,5):
            loop = asyncio.get_event_loop()
            new_action, deepth, rollouts = loop.run_until_complete(gnn.montecarlo_alg.get_action(M, 5, 2, x, y, gnn.Simulation_score()))
            new_action.execute(M)
            acc, _ = M.gradient_descent(x, y, epochs, lr_scheduler, True)
        acc, _ = M.gradient_descent(x, y, epochs, lr_scheduler, True)
        self.assertEqual(acc >= 0.01, True)

if __name__ == '__main__':
    unittest.main()
