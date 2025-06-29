import asyncio
import sys
import os
import tempfile
sys.path.append('.')
sys.path.append('../')
import growingnn as gnn
import numpy as np
import unittest
import random
from testSuite import mode

shape = 20
epochs = 5

class TestingTrain(unittest.TestCase):

    def setUp(self):
        global mode
        if mode == 'cpu':
            gnn.switch_to_cpu()
        elif mode == 'gpu':
            gnn.switch_to_gpu()
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, "testimage.html")

    def tearDown(self):
        # Clean up the temporary directory
        try:
            for root, dirs, files in os.walk(self.test_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self.test_dir)
        except Exception as e:
            print(f"Warning: Could not clean up temporary directory: {str(e)}")

    def test_drawing_graphs(self):
        success = False
        try:
            M = gnn.structure.Model(shape, shape, 2, gnn.structure.Loss.multiclass_cross_entropy, gnn.structure.Activations.Sigmoid, 1)
            gnn.painter.draw(M, self.test_file)
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
        self.assertEqual(acc >= 0.3, True)

    def test_simple_SGD_train(self):
        M = gnn.structure.Model(shape,shape,2, gnn.structure.Loss.multiclass_cross_entropy, gnn.structure.Activations.Sigmoid, 1, gnn.optimizers.SGDOptimizer())
        x = np.random.rand(shape, shape)
        y = np.random.randint(2, size=(shape,))
        lr_scheduler = gnn.structure.LearningRateScheduler(gnn.structure.LearningRateScheduler.PROGRESIVE, 0.03, 0.8)
        gnn.painter.draw(M, "input_test.html")
        acc, _ = M.gradient_descent(x, y, epochs, lr_scheduler)
        self.assertEqual(acc >= 0.3, True)

    def test_simple_Adam_train(self):
        M = gnn.structure.Model(shape,shape,2, gnn.structure.Loss.multiclass_cross_entropy, gnn.structure.Activations.Sigmoid, 1, gnn.optimizers.AdamOptimizer())
        x = np.random.rand(shape, shape)
        y = np.random.randint(2, size=(shape,))
        lr_scheduler = gnn.structure.LearningRateScheduler(gnn.structure.LearningRateScheduler.PROGRESIVE, 0.03, 0.8)
        gnn.painter.draw(M, "input_test.html")
        acc, _ = M.gradient_descent(x, y, epochs, lr_scheduler)
        self.assertEqual(acc >= 0.3, True)

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
        
        # Create a new event loop for the test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            action, deepth, rollouts = loop.run_until_complete(gnn.montecarlo_alg.get_action(M, 5, 2, x, y, gnn.Simulation_score()))
            acc, _ = M.gradient_descent(x, y, epochs, lr_scheduler, True)
            print("test_montecarlo result acc: ", acc)
            self.assertEqual(acc >= 0.01, True)
        finally:
            loop.close()

    def test_MCTS_simulation(self):
        M = gnn.structure.Model(shape,shape,2, gnn.structure.Loss.multiclass_cross_entropy, gnn.structure.Activations.Sigmoid, 1)
        x = np.random.rand(shape, shape)
        y = np.random.randint(2, size=(shape,))
        lr_scheduler = gnn.structure.LearningRateScheduler(gnn.structure.LearningRateScheduler.PROGRESIVE, 0.03, 0.8)
        acc = M.gradient_descent(x, y, epochs, lr_scheduler, True)
        
        # Create a new event loop for the test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            for i in range(0,5):
                new_action, deepth, rollouts = loop.run_until_complete(gnn.montecarlo_alg.get_action(M, 5, 2, x, y, gnn.Simulation_score()))
                new_action.execute(M)
                acc, _ = M.gradient_descent(x, y, epochs, lr_scheduler, True)
            acc, _ = M.gradient_descent(x, y, epochs, lr_scheduler, True)
            self.assertEqual(acc >= 0.01, True)
        finally:
            loop.close()

if __name__ == '__main__':
    unittest.main()
