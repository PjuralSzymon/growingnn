import sys
sys.path.append('.')
sys.path.append('../')
import growingnn as gnn
import numpy as np
import unittest
import random
import time
import asyncio
from testSuite import mode

shape = 20
epochs = 1

class TestingSimulationScore(unittest.TestCase):

    def setUp(self):
        global mode
        if mode == 'cpu':
            gnn.switch_to_cpu()
        elif mode == 'gpu':
            gnn.switch_to_gpu()
            
        self.shape = shape
        self.epochs = epochs
        self.M = gnn.structure.Model(self.shape, self.shape, 2, gnn.structure.Loss.multiclass_cross_entropy, gnn.structure.Activations.Sigmoid, 1)
        self.x = np.random.rand(self.shape, self.shape)
        self.y = np.random.randint(2, size=(self.shape,))
        self.lr_scheduler = gnn.structure.LearningRateScheduler(gnn.structure.LearningRateScheduler.PROGRESIVE, 0.03, 0.8)
        
    def test_basic_simulation_score(self):
        """Test basic functionality of simulation score"""
        # Apply some random actions to the model
        for i in range(0, 5):
            all_actions = gnn.action.Action.generate_all_actions(self.M)
            if all_actions:
                new_action = random.choice(all_actions)        
                new_action.execute(self.M)
        
        # Train the model
        self.M.gradient_descent(self.x, self.y, 1, self.lr_scheduler, True)
        
        # Create simulation score with default parameters
        simulation_score = gnn.Simulation_score(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        score = simulation_score.scoreFun(self.M, 10, self.x, self.y)
        
        # Verify score is a number
        self.assertIsInstance(score, (int, float))
        print("Basic simulation score:", score)
        
    def test_simulation_score_parameters(self):
        """Test simulation score with different parameter values"""
        # Apply some random actions to the model
        for i in range(0, 5):
            all_actions = gnn.action.Action.generate_all_actions(self.M)
            if all_actions:
                new_action = random.choice(all_actions)        
                new_action.execute(self.M)
        
        # Train the model
        self.M.gradient_descent(self.x, self.y, 1, self.lr_scheduler, True)
        
        # Test with different parameter values
        params = [
            (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
            (0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9),
            (0.5, 0.7, 0.3, 0.6, 0.4, 0.8, 0.2, 0.9, 0.1, 0.5)
        ]
        
        scores = []
        for param_set in params:
            simulation_score = gnn.Simulation_score(*param_set)
            score = simulation_score.scoreFun(self.M, 10, self.x, self.y)
            scores.append(score)
            print(f"Simulation score with params {param_set}: {score}")
        
        # Verify scores are different for different parameters
        self.assertTrue(len(set(scores)) > 1, "All scores are the same despite different parameters")
        
    def test_monte_carlo_algorithm(self):
        """Test the Monte Carlo Tree Search algorithm"""
        # Apply some random actions to the model
        for i in range(0, 3):
            all_actions = gnn.action.Action.generate_all_actions(self.M)
            if all_actions:
                new_action = random.choice(all_actions)        
                new_action.execute(self.M)
        
        # Create simulation score
        simulation_score = gnn.Simulation_score(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        
        # Run Monte Carlo algorithm with a short time limit
        start_time = time.time()
        
        # Use asyncio to run the async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        action, depth, rollouts = loop.run_until_complete(
            gnn.montecarlo_alg.get_action(
                self.M, 
                max_time_for_dec=2,  # 2 seconds time limit
                epochs=5,
                X_train=self.x,
                Y_train=self.y,
                simulation_score=simulation_score
            )
        )
        loop.close()
        
        elapsed_time = time.time() - start_time
        
        # Verify the algorithm returns valid results
        self.assertIsNotNone(action, "Monte Carlo algorithm returned None action")
        self.assertIsInstance(depth, int, "Depth should be an integer")
        self.assertIsInstance(rollouts, int, "Rollouts should be an integer")
        self.assertTrue(rollouts > 0, "Should have performed at least one rollout")
        self.assertTrue(elapsed_time >= 2, "Algorithm should run for at least the specified time")
        
        print(f"Monte Carlo algorithm: depth={depth}, rollouts={rollouts}, time={elapsed_time:.2f}s")
        
    def test_greedy_algorithm(self):
        """Test the Greedy algorithm"""
        # Apply some random actions to the model
        for i in range(0, 3):
            all_actions = gnn.action.Action.generate_all_actions(self.M)
            if all_actions:
                new_action = random.choice(all_actions)        
                new_action.execute(self.M)
        
        # Create simulation score
        simulation_score = gnn.Simulation_score(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        
        # Run Greedy algorithm with a short time limit
        start_time = time.time()
        
        # Use asyncio to run the async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        action, depth, rollouts = loop.run_until_complete(
            gnn.greedy_alg.get_action(
                self.M, 
                max_time_for_dec=2,  # 2 seconds time limit
                epochs=5,
                X_train=self.x,
                Y_train=self.y,
                simulation_score=simulation_score
            )
        )
        loop.close()
        
        elapsed_time = time.time() - start_time
        
        # Verify the algorithm returns valid results
        self.assertIsNotNone(action, "Greedy algorithm returned None action")
        self.assertIsInstance(depth, int, "Depth should be an integer")
        self.assertIsInstance(rollouts, int, "Rollouts should be an integer")
        self.assertTrue(rollouts > 0, "Should have performed at least one rollout")
        
        print(f"Greedy algorithm: depth={depth}, rollouts={rollouts}, time={elapsed_time:.2f}s")
        
    def test_random_algorithm(self):
        """Test the Random algorithm"""
        # Apply some random actions to the model
        for i in range(0, 3):
            all_actions = gnn.action.Action.generate_all_actions(self.M)
            if all_actions:
                new_action = random.choice(all_actions)        
                new_action.execute(self.M)
        
        # Create simulation score
        simulation_score = gnn.Simulation_score(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        
        # Run Random algorithm
        # Use asyncio to run the async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        action, depth, rollouts = loop.run_until_complete(
            gnn.random_alg.get_action(
                self.M, 
                max_time_for_dec=2,
                epochs=5,
                X_train=self.x,
                Y_train=self.y,
                simulation_score=simulation_score
            )
        )
        loop.close()
        
        # Verify the algorithm returns valid results
        self.assertIsNotNone(action, "Random algorithm returned None action")
        self.assertEqual(depth, 0, "Random algorithm should have depth 0")
        self.assertEqual(rollouts, 0, "Random algorithm should have 0 rollouts")
        
        print(f"Random algorithm: action={action}")
        
    def test_algorithm_comparison(self):
        """Compare the performance of different algorithms"""
        # Apply some random actions to the model
        for i in range(0, 3):
            all_actions = gnn.action.Action.generate_all_actions(self.M)
            if all_actions:
                new_action = random.choice(all_actions)        
                new_action.execute(self.M)
        
        # Create simulation score
        simulation_score = gnn.Simulation_score(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        
        # Run all algorithms with the same time limit
        time_limit = 3
        
        # Create event loop for async functions
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Monte Carlo
        start_time = time.time()
        mc_action, mc_depth, mc_rollouts = loop.run_until_complete(
            gnn.montecarlo_alg.get_action(
                self.M.deepcopy(), 
                max_time_for_dec=time_limit,
                epochs=5,
                X_train=self.x,
                Y_train=self.y,
                simulation_score=simulation_score
            )
        )
        mc_time = time.time() - start_time
        
        # Greedy
        start_time = time.time()
        greedy_action, greedy_depth, greedy_rollouts = loop.run_until_complete(
            gnn.greedy_alg.get_action(
                self.M.deepcopy(), 
                max_time_for_dec=time_limit,
                epochs=5,
                X_train=self.x,
                Y_train=self.y,
                simulation_score=simulation_score
            )
        )
        greedy_time = time.time() - start_time
        
        # Random
        start_time = time.time()
        random_action, random_depth, random_rollouts = loop.run_until_complete(
            gnn.random_alg.get_action(
                self.M.deepcopy(), 
                max_time_for_dec=time_limit,
                epochs=5,
                X_train=self.x,
                Y_train=self.y,
                simulation_score=simulation_score
            )
        )
        random_time = time.time() - start_time
        
        loop.close()
        
        # Print comparison
        print("\nAlgorithm Comparison:")
        print(f"Monte Carlo: depth={mc_depth}, rollouts={mc_rollouts}, time={mc_time:.2f}s")
        print(f"Greedy: depth={greedy_depth}, rollouts={greedy_rollouts}, time={greedy_time:.2f}s")
        print(f"Random: depth={random_depth}, rollouts={random_rollouts}, time={random_time:.2f}s")
        
        # Verify that Monte Carlo and Greedy perform more rollouts than Random
        self.assertTrue(mc_rollouts > random_rollouts, "Monte Carlo should perform more rollouts than Random")
        self.assertTrue(greedy_rollouts > random_rollouts, "Greedy should perform more rollouts than Random")
        
    def test_simulation_score_with_different_models(self):
        """Test simulation score with different model architectures"""
        # Create different model architectures
        models = [
            gnn.structure.Model(self.shape, self.shape, 2, gnn.structure.Loss.multiclass_cross_entropy, gnn.structure.Activations.Sigmoid, 1),
            gnn.structure.Model(self.shape, self.shape*2, 2, gnn.structure.Loss.multiclass_cross_entropy, gnn.structure.Activations.ReLu, 1),
            gnn.structure.Model(self.shape, self.shape//2, 2, gnn.structure.Loss.multiclass_cross_entropy, gnn.structure.Activations.Tanh, 1)
        ]
        
        # Create simulation score
        simulation_score = gnn.Simulation_score(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        
        scores = []
        for i, model in enumerate(models):
            # Apply some random actions to the model
            for j in range(0, 3):
                all_actions = gnn.action.Action.generate_all_actions(model)
                if all_actions:
                    new_action = random.choice(all_actions)        
                    new_action.execute(model)
            
            # Train the model
            model.gradient_descent(self.x, self.y, 1, self.lr_scheduler, True)
            
            # Calculate score
            score = simulation_score.scoreFun(model, 10, self.x, self.y)
            scores.append(score)
            print(f"Model {i+1} score: {score}")
        
        # Verify scores are different for different models
        self.assertTrue(len(set(scores)) > 1, "All scores are the same despite different models")

if __name__ == '__main__':
    unittest.main()
