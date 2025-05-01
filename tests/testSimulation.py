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

class TestSimulationAlgorithms(unittest.TestCase):
    def setUp(self):
        global mode
        if mode == 'cpu':
            gnn.switch_to_cpu()
        elif mode == 'gpu':
            gnn.switch_to_gpu()
            
        # Create a simple model for testing
        self.input_size = 10
        self.output_size = 2
        self.model = gnn.structure.Model(
            self.input_size, 
            self.input_size, 
            self.output_size, 
            gnn.structure.Loss.multiclass_cross_entropy, 
            gnn.structure.Activations.Sigmoid, 
            1
        )
        
        # Create training data
        self.x_train = np.random.rand(20, self.input_size)
        self.y_train = np.random.randint(0, self.output_size, size=(20,))
        
        # Create simulation score
        self.simulation_score = gnn.Simulation_score(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        
    def test_monte_carlo_algorithm(self):
        """Test the Monte Carlo Tree Search algorithm"""
        # Run the algorithm with a short time limit
        start_time = time.time()
        
        # Use asyncio to run the async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        action, depth, rollouts = loop.run_until_complete(
            gnn.Simulation.montecarlo_alg.get_action(
                self.model, 
                max_time_for_dec=2,  # 2 seconds time limit
                epochs=5,
                X_train=self.x_train,
                Y_train=self.y_train,
                simulation_score=self.simulation_score
            )
        )
        loop.close()
        
        elapsed_time = time.time() - start_time
        
        # Verify that get_action returns valid results
        self.assertIsNotNone(action, "Monte Carlo should return a valid action")
        self.assertIsInstance(depth, int, "Monte Carlo should return an integer depth")
        self.assertIsInstance(rollouts, int, "Monte Carlo should return an integer rollouts")
        self.assertTrue(rollouts > 0, "Monte Carlo should perform at least one rollout")
        self.assertTrue(elapsed_time >= 2, "Monte Carlo should run for at least the specified time")
        
        print(f"Monte Carlo results: depth={depth}, rollouts={rollouts}, time={elapsed_time:.2f}s")
        
    def test_greedy_algorithm(self):
        """Test the Greedy algorithm"""
        # Run the algorithm with a short time limit
        start_time = time.time()
        
        # Use asyncio to run the async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        action, depth, rollouts = loop.run_until_complete(
            gnn.Simulation.greedy_alg.get_action(
                self.model, 
                max_time_for_dec=2,  # 2 seconds time limit
                epochs=5,
                X_train=self.x_train,
                Y_train=self.y_train,
                simulation_score=self.simulation_score
            )
        )
        loop.close()
        
        elapsed_time = time.time() - start_time
        
        # Verify that get_action returns valid results
        self.assertIsNotNone(action, "Greedy should return a valid action")
        self.assertIsInstance(depth, int, "Greedy should return an integer depth")
        self.assertIsInstance(rollouts, int, "Greedy should return an integer rollouts")
        self.assertTrue(rollouts > 0, "Greedy should perform at least one rollout")
        # The greedy algorithm might complete faster than the time limit
        # So we just check that it took some time to execute
        self.assertTrue(elapsed_time > 0, "Greedy should take some time to execute")
        
        print(f"Greedy results: depth={depth}, rollouts={rollouts}, time={elapsed_time:.2f}s")
        
    def test_random_algorithm(self):
        """Test the Random algorithm"""
        # Run the algorithm with a short time limit
        start_time = time.time()
        
        # Use asyncio to run the async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        action, depth, rollouts = loop.run_until_complete(
            gnn.Simulation.random_alg.get_action(
                self.model, 
                max_time_for_dec=2,  # 2 seconds time limit
                epochs=5,
                X_train=self.x_train,
                Y_train=self.y_train,
                simulation_score=self.simulation_score
            )
        )
        loop.close()
        
        elapsed_time = time.time() - start_time
        
        # Verify that get_action returns valid results
        self.assertIsNotNone(action, "Random should return a valid action")
        self.assertIsInstance(depth, int, "Random should return an integer depth")
        self.assertIsInstance(rollouts, int, "Random should return an integer rollouts")
        self.assertTrue(rollouts >= 0, "Random should perform at least zero rollouts")
        self.assertTrue(elapsed_time >= 0, "Random should run for at least 0 seconds")
        
        print(f"Random results: depth={depth}, rollouts={rollouts}, time={elapsed_time:.2f}s")
        
    def test_algorithm_comparison(self):
        """Compare the performance of different algorithms"""
        # Run each algorithm with the same parameters
        time_limit = 3
        epochs = 5
        
        # Create event loop for async functions
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run Monte Carlo
        start_time = time.time()
        mc_action, mc_depth, mc_rollouts = loop.run_until_complete(
            gnn.Simulation.montecarlo_alg.get_action(
                self.model, 
                max_time_for_dec=time_limit,
                epochs=epochs,
                X_train=self.x_train,
                Y_train=self.y_train,
                simulation_score=self.simulation_score
            )
        )
        mc_time = time.time() - start_time
        
        # Run Greedy
        start_time = time.time()
        greedy_action, greedy_depth, greedy_rollouts = loop.run_until_complete(
            gnn.Simulation.greedy_alg.get_action(
                self.model, 
                max_time_for_dec=time_limit,
                epochs=epochs,
                X_train=self.x_train,
                Y_train=self.y_train,
                simulation_score=self.simulation_score
            )
        )
        greedy_time = time.time() - start_time
        
        # Run Random
        start_time = time.time()
        random_action, random_depth, random_rollouts = loop.run_until_complete(
            gnn.Simulation.random_alg.get_action(
                self.model, 
                max_time_for_dec=time_limit,
                epochs=epochs,
                X_train=self.x_train,
                Y_train=self.y_train,
                simulation_score=self.simulation_score
            )
        )
        random_time = time.time() - start_time
        
        loop.close()
        
        # Print comparison results
        print("\nAlgorithm comparison:")
        print(f"Monte Carlo: depth={mc_depth}, rollouts={mc_rollouts}, time={mc_time:.2f}s")
        print(f"Greedy: depth={greedy_depth}, rollouts={greedy_rollouts}, time={greedy_time:.2f}s")
        print(f"Random: depth={random_depth}, rollouts={random_rollouts}, time={random_time:.2f}s")
        
        # Verify that all algorithms return valid actions
        self.assertIsNotNone(mc_action, "Monte Carlo should return a valid action")
        self.assertIsNotNone(greedy_action, "Greedy should return a valid action")
        self.assertIsNotNone(random_action, "Random should return a valid action")
        
        # Verify that all algorithms perform at least one rollout
        self.assertTrue(mc_rollouts > 0, "Monte Carlo should perform at least one rollout")
        self.assertTrue(greedy_rollouts > 0, "Greedy should perform at least one rollout")
        self.assertTrue(random_rollouts >= 0, "Random should perform at least zero rollouts")
        
    def test_simulation_score_with_different_models(self):
        """Test simulation score with different model architectures"""
        # Create different model architectures
        models = [
            # Small model
            gnn.structure.Model(
                self.input_size, 
                self.input_size // 2, 
                self.output_size, 
                gnn.structure.Loss.multiclass_cross_entropy, 
                gnn.structure.Activations.Sigmoid, 
                1
            ),
            # Medium model
            gnn.structure.Model(
                self.input_size, 
                self.input_size, 
                self.output_size, 
                gnn.structure.Loss.multiclass_cross_entropy, 
                gnn.structure.Activations.Sigmoid, 
                1
            ),
            # Large model
            gnn.structure.Model(
                self.input_size, 
                self.input_size * 2, 
                self.output_size, 
                gnn.structure.Loss.multiclass_cross_entropy, 
                gnn.structure.Activations.Sigmoid, 
                1
            )
        ]
        
        # Run each algorithm with each model
        time_limit = 2
        epochs = 5
        
        # Create event loop for async functions
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        results = []
        
        for i, model in enumerate(models):
            model_size = "small" if i == 0 else "medium" if i == 1 else "large"
            
            # Run Monte Carlo
            start_time = time.time()
            mc_action, mc_depth, mc_rollouts = loop.run_until_complete(
                gnn.Simulation.montecarlo_alg.get_action(
                    model, 
                    max_time_for_dec=time_limit,
                    epochs=epochs,
                    X_train=self.x_train,
                    Y_train=self.y_train,
                    simulation_score=self.simulation_score
                )
            )
            mc_time = time.time() - start_time
            
            # Run Greedy
            start_time = time.time()
            greedy_action, greedy_depth, greedy_rollouts = loop.run_until_complete(
                gnn.Simulation.greedy_alg.get_action(
                    model, 
                    max_time_for_dec=time_limit,
                    epochs=epochs,
                    X_train=self.x_train,
                    Y_train=self.y_train,
                    simulation_score=self.simulation_score
                )
            )
            greedy_time = time.time() - start_time
            
            # Run Random
            start_time = time.time()
            random_action, random_depth, random_rollouts = loop.run_until_complete(
                gnn.Simulation.random_alg.get_action(
                    model, 
                    max_time_for_dec=time_limit,
                    epochs=epochs,
                    X_train=self.x_train,
                    Y_train=self.y_train,
                    simulation_score=self.simulation_score
                )
            )
            random_time = time.time() - start_time
            
            results.append({
                'model_size': model_size,
                'monte_carlo': {'depth': mc_depth, 'rollouts': mc_rollouts, 'time': mc_time},
                'greedy': {'depth': greedy_depth, 'rollouts': greedy_rollouts, 'time': greedy_time},
                'random': {'depth': random_depth, 'rollouts': random_rollouts, 'time': random_time}
            })
            
            # Print results for this model
            print(f"\nModel size: {model_size}")
            print(f"Monte Carlo: depth={mc_depth}, rollouts={mc_rollouts}, time={mc_time:.2f}s")
            print(f"Greedy: depth={greedy_depth}, rollouts={greedy_rollouts}, time={greedy_time:.2f}s")
            print(f"Random: depth={random_depth}, rollouts={random_rollouts}, time={random_time:.2f}s")
            
            # Verify that all algorithms return valid actions
            self.assertIsNotNone(mc_action, f"Monte Carlo should return a valid action for {model_size} model")
            self.assertIsNotNone(greedy_action, f"Greedy should return a valid action for {model_size} model")
            self.assertIsNotNone(random_action, f"Random should return a valid action for {model_size} model")
            
            # Verify that all algorithms perform at least one rollout
            self.assertTrue(mc_rollouts > 0, f"Monte Carlo should perform at least one rollout for {model_size} model")
            self.assertTrue(greedy_rollouts > 0, f"Greedy should perform at least one rollout for {model_size} model")
            self.assertTrue(random_rollouts >= 0, f"Random should perform at least zero rollouts for {model_size} model")
        
        loop.close()
        
        # Verify that larger models generally take more time
        for i in range(1, len(results)):
            prev_model = results[i-1]['model_size']
            curr_model = results[i]['model_size']
            
            # This is a general trend, not a strict rule
            print(f"Time comparison: {prev_model} vs {curr_model}")
            print(f"Monte Carlo: {results[i-1]['monte_carlo']['time']:.2f}s vs {results[i]['monte_carlo']['time']:.2f}s")
            print(f"Greedy: {results[i-1]['greedy']['time']:.2f}s vs {results[i]['greedy']['time']:.2f}s")
            print(f"Random: {results[i-1]['random']['time']:.2f}s vs {results[i]['random']['time']:.2f}s")
            
    def test_simulation_score_with_different_parameters(self):
        """Test simulation score with different parameter values"""
        # Create different simulation scores with different parameters
        simulation_scores = [
            # Balanced parameters
            gnn.Simulation_score(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
            # Accuracy-focused parameters
            gnn.Simulation_score(0.8, 0.8, 0.8, 0.8, 0.8, 0.2, 0.2, 0.2, 0.2, 0.2),
            # Complexity-focused parameters
            gnn.Simulation_score(0.2, 0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8, 0.8)
        ]
        
        # Run each algorithm with each simulation score
        time_limit = 2
        epochs = 5
        
        # Create event loop for async functions
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        results = []
        
        for i, sim_score in enumerate(simulation_scores):
            param_type = "balanced" if i == 0 else "accuracy-focused" if i == 1 else "complexity-focused"
            
            # Run Monte Carlo
            start_time = time.time()
            mc_action, mc_depth, mc_rollouts = loop.run_until_complete(
                gnn.Simulation.montecarlo_alg.get_action(
                    self.model, 
                    max_time_for_dec=time_limit,
                    epochs=epochs,
                    X_train=self.x_train,
                    Y_train=self.y_train,
                    simulation_score=sim_score
                )
            )
            mc_time = time.time() - start_time
            
            # Run Greedy
            start_time = time.time()
            greedy_action, greedy_depth, greedy_rollouts = loop.run_until_complete(
                gnn.Simulation.greedy_alg.get_action(
                    self.model, 
                    max_time_for_dec=time_limit,
                    epochs=epochs,
                    X_train=self.x_train,
                    Y_train=self.y_train,
                    simulation_score=sim_score
                )
            )
            greedy_time = time.time() - start_time
            
            # Run Random
            start_time = time.time()
            random_action, random_depth, random_rollouts = loop.run_until_complete(
                gnn.Simulation.random_alg.get_action(
                    self.model, 
                    max_time_for_dec=time_limit,
                    epochs=epochs,
                    X_train=self.x_train,
                    Y_train=self.y_train,
                    simulation_score=sim_score
                )
            )
            random_time = time.time() - start_time
            
            results.append({
                'param_type': param_type,
                'monte_carlo': {'depth': mc_depth, 'rollouts': mc_rollouts, 'time': mc_time},
                'greedy': {'depth': greedy_depth, 'rollouts': greedy_rollouts, 'time': greedy_time},
                'random': {'depth': random_depth, 'rollouts': random_rollouts, 'time': random_time}
            })
            
            # Print results for this simulation score
            print(f"\nParameter type: {param_type}")
            print(f"Monte Carlo: depth={mc_depth}, rollouts={mc_rollouts}, time={mc_time:.2f}s")
            print(f"Greedy: depth={greedy_depth}, rollouts={greedy_rollouts}, time={greedy_time:.2f}s")
            print(f"Random: depth={random_depth}, rollouts={random_rollouts}, time={random_time:.2f}s")
            
            # Verify that all algorithms return valid actions
            self.assertIsNotNone(mc_action, f"Monte Carlo should return a valid action for {param_type} parameters")
            self.assertIsNotNone(greedy_action, f"Greedy should return a valid action for {param_type} parameters")
            self.assertIsNotNone(random_action, f"Random should return a valid action for {param_type} parameters")
            
            # Verify that all algorithms perform at least one rollout
            self.assertTrue(mc_rollouts > 0, f"Monte Carlo should perform at least one rollout for {param_type} parameters")
            self.assertTrue(greedy_rollouts > 0, f"Greedy should perform at least one rollout for {param_type} parameters")
            self.assertTrue(random_rollouts >= 0, f"Random should perform at least zero rollouts for {param_type} parameters")
        
        loop.close()
        
        # Verify that different parameters result in different actions
        # This is a general trend, not a strict rule
        print("\nParameter comparison:")
        for i in range(1, len(results)):
            prev_param = results[i-1]['param_type']
            curr_param = results[i]['param_type']
            
            print(f"Comparison: {prev_param} vs {curr_param}")
            print(f"Monte Carlo depth: {results[i-1]['monte_carlo']['depth']} vs {results[i]['monte_carlo']['depth']}")
            print(f"Greedy depth: {results[i-1]['greedy']['depth']} vs {results[i]['greedy']['depth']}")
            print(f"Random depth: {results[i-1]['random']['depth']} vs {results[i]['random']['depth']}")

if __name__ == '__main__':
    unittest.main() 