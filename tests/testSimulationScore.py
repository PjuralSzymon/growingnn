import sys
sys.path.append('.')
sys.path.append('../')
import growingnn as gnn
import numpy as np
import unittest
import random
from testSuite import mode

shape = 20
epochs = 1

class TestingTrain(unittest.TestCase):

    def test_actions(self):
        M = gnn.structure.Model(shape,shape,2, gnn.structure.Loss.multiclass_cross_entropy, gnn.structure.Activations.Sigmoid, 1)
        x = np.random.rand(shape, shape)
        y = np.random.randint(2, size=(shape,))
        lr_scheduler = gnn.structure.LearningRateScheduler(gnn.structure.LearningRateScheduler.PROGRESIVE, 0.03, 0.8)
        for i in range(0,10):
            all_actions = gnn.action.Action.generate_all_actions(M)
            new_action = random.choice(all_actions)        
            new_action.execute(M)
        M.gradient_descent(x, y, 1, lr_scheduler, True)
        simulation_score = gnn.Simulation_score(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        score = simulation_score.scoreFun(M, 10, x, y)
        print("score: ", score)
        

if __name__ == '__main__':
    unittest.main()
