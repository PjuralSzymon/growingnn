import sys
sys.path.append('.')
sys.path.append('../')
import growingnn as gnn
import numpy as np
import time
import unittest
import pandas as pd

shape = 10

class TestingTrain(unittest.TestCase):

    def test_base_structure(self):
        print("start")
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
        acc, _ = M.gradient_descent(x, y, 10, lr_scheduler)
        self.assertEqual(acc >= 0.4, True)

if __name__ == '__main__':
    unittest.main()
