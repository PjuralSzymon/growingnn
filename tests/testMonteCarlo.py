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

class TestMonteCarloTreeSearch(unittest.TestCase):
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
        
    def test_tree_node_creation(self):
        """Test creation of TreeNode objects"""
        # Create a TreeNode
        node = gnn.montecarlo_alg.TreeNode(
            _parent=None,
            _action=None,
            _M=self.model,
            _epochs=5,
            _X_train=self.x_train,
            _Y_train=self.y_train,
            _simulation_score=self.simulation_score
        )
        
        # Verify node properties
        self.assertIsNone(node.parent, "Parent should be None for root node")
        self.assertIsNone(node.action, "Action should be None for root node")
        self.assertEqual(node.M, self.model, "Model should match the input model")
        self.assertEqual(node.epochs, 5, "Epochs should match the input value")
        self.assertTrue(np.array_equal(node.X_train, self.x_train), "X_train should match the input data")
        self.assertTrue(np.array_equal(node.Y_train, self.y_train), "Y_train should match the input data")
        self.assertEqual(node.simulation_score, self.simulation_score, "Simulation score should match the input")
        self.assertEqual(node.value, 0, "Initial value should be 0")
        self.assertEqual(node.visit_counter, 0, "Initial visit counter should be 0")
        self.assertEqual(len(node.childNodes), 0, "Initial child nodes should be empty")
        
    def test_tree_node_expansion(self):
        """Test expansion of TreeNode"""
        # Create a TreeNode
        node = gnn.montecarlo_alg.TreeNode(
            _parent=None,
            _action=None,
            _M=self.model,
            _epochs=5,
            _X_train=self.x_train,
            _Y_train=self.y_train,
            _simulation_score=self.simulation_score
        )
        
        # Expand the node
        node.expand()
        
        # Verify that child nodes were created
        self.assertTrue(len(node.childNodes) > 0, "Node should have child nodes after expansion")
        
        # Verify that each child node has the correct properties
        for child in node.childNodes:
            self.assertEqual(child.parent, node, "Child parent should be the original node")
            self.assertIsNotNone(child.action, "Child action should not be None")
            self.assertNotEqual(child.M, self.model, "Child model should be different from parent model")
            self.assertEqual(child.epochs, 5, "Child epochs should match parent")
            self.assertTrue(np.array_equal(child.X_train, self.x_train), "Child X_train should match parent")
            self.assertTrue(np.array_equal(child.Y_train, self.y_train), "Child Y_train should match parent")
            self.assertEqual(child.simulation_score, self.simulation_score, "Child simulation score should match parent")
            self.assertEqual(child.value, 0, "Initial child value should be 0")
            self.assertEqual(child.visit_counter, 0, "Initial child visit counter should be 0")
            self.assertEqual(len(child.childNodes), 0, "Initial child nodes should be empty")
            
    def test_tree_node_rollout(self):
        """Test rollout of TreeNode"""
        # Create a TreeNode
        node = gnn.montecarlo_alg.TreeNode(
            _parent=None,
            _action=None,
            _M=self.model,
            _epochs=5,
            _X_train=self.x_train,
            _Y_train=self.y_train,
            _simulation_score=self.simulation_score
        )
        
        # Perform rollout
        score = node.rollout()
        
        # Verify that rollout returns a score
        self.assertIsInstance(score, (int, float), "Rollout should return a numeric score")
        print(f"Rollout score: {score}")
        
    def test_tree_node_get_best_child(self):
        """Test getting the best child from a TreeNode"""
        # Create a TreeNode
        node = gnn.montecarlo_alg.TreeNode(
            _parent=None,
            _action=None,
            _M=self.model,
            _epochs=5,
            _X_train=self.x_train,
            _Y_train=self.y_train,
            _simulation_score=self.simulation_score
        )
        
        # Expand the node
        node.expand()
        
        # If there are no child nodes, the test should be skipped
        if len(node.childNodes) == 0:
            self.skipTest("No child nodes available for testing get_best_child")
            
        # Set different values for child nodes and ensure visit_counter is at least 1
        # to avoid math domain error in UCB1 calculation
        for i, child in enumerate(node.childNodes):
            child.value = i
            child.visit_counter = 1  # Ensure visit_counter is at least 1
            
        # Set parent visit_counter to a value greater than 0
        node.visit_counter = 10
        
        # Get the best child
        best_child = node.get_best_child()
        
        # Verify that the best child is the one with the highest value
        self.assertEqual(best_child.value, len(node.childNodes) - 1, "Best child should have the highest value")
        
    def test_tree_node_is_leaf(self):
        """Test is_leaf method of TreeNode"""
        # Create a TreeNode
        node = gnn.montecarlo_alg.TreeNode(
            _parent=None,
            _action=None,
            _M=self.model,
            _epochs=5,
            _X_train=self.x_train,
            _Y_train=self.y_train,
            _simulation_score=self.simulation_score
        )
        
        # Verify that a new node is a leaf
        self.assertTrue(node.is_leaf(), "New node should be a leaf")
        
        # Expand the node
        node.expand()
        
        # Verify that the node is no longer a leaf
        self.assertFalse(node.is_leaf(), "Expanded node should not be a leaf")
        
    def test_tree_node_get_depth(self):
        """Test get_depth method of TreeNode"""
        # Create a root node
        root = gnn.montecarlo_alg.TreeNode(
            _parent=None,
            _action=None,
            _M=self.model,
            _epochs=5,
            _X_train=self.x_train,
            _Y_train=self.y_train,
            _simulation_score=self.simulation_score
        )
        
        # Verify that a new node has depth 1
        self.assertEqual(root.get_depth(), 1, "New node should have depth 1")
        
        # Expand the root node
        root.expand()
        
        # If there are no child nodes, the test should be skipped
        if len(root.childNodes) == 0:
            self.skipTest("No child nodes available for testing get_depth")
            
        # Verify that the root node has depth 2
        self.assertEqual(root.get_depth(), 2, "Root node with children should have depth 2")
        
        # Expand one of the child nodes
        child = root.childNodes[0]
        child.expand()
        
        # Verify that the root node has depth 3
        self.assertEqual(root.get_depth(), 3, "Root node with grandchildren should have depth 3")
        
    def test_simulate_function(self):
        """Test the simulate function"""
        # Create a root node
        root = gnn.montecarlo_alg.TreeNode(
            _parent=None,
            _action=None,
            _M=self.model,
            _epochs=5,
            _X_train=self.x_train,
            _Y_train=self.y_train,
            _simulation_score=self.simulation_score
        )
        
        # Simulate from the root node
        value, depth, rollouts = gnn.montecarlo_alg.simulate(root)
        
        # Verify that simulate returns valid results
        self.assertIsInstance(value, (int, float), "Simulate should return a numeric value")
        self.assertIsInstance(depth, int, "Simulate should return an integer depth")
        self.assertIsInstance(rollouts, int, "Simulate should return an integer rollouts")
        self.assertTrue(rollouts > 0, "Simulate should perform at least one rollout")
        
        print(f"Simulate results: value={value}, depth={depth}, rollouts={rollouts}")
        
    def test_get_action_function(self):
        """Test the get_action function"""
        # Run get_action with a short time limit
        start_time = time.time()
        
        # Use asyncio to run the async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        action, depth, rollouts = loop.run_until_complete(
            gnn.montecarlo_alg.get_action(
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
        self.assertIsNotNone(action, "get_action should return a valid action")
        self.assertIsInstance(depth, int, "get_action should return an integer depth")
        self.assertIsInstance(rollouts, int, "get_action should return an integer rollouts")
        self.assertTrue(rollouts > 0, "get_action should perform at least one rollout")
        self.assertTrue(elapsed_time >= 2, "get_action should run for at least the specified time")
        
        print(f"get_action results: depth={depth}, rollouts={rollouts}, time={elapsed_time:.2f}s")
        
    def test_get_action_with_different_time_limits(self):
        """Test get_action with different time limits"""
        time_limits = [1, 3, 5]
        results = []
        
        # Create event loop for async functions
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        for time_limit in time_limits:
            start_time = time.time()
            action, depth, rollouts = loop.run_until_complete(
                gnn.montecarlo_alg.get_action(
                    self.model, 
                    max_time_for_dec=time_limit,
                    epochs=5,
                    X_train=self.x_train,
                    Y_train=self.y_train,
                    simulation_score=self.simulation_score
                )
            )
            elapsed_time = time.time() - start_time
            
            results.append((time_limit, depth, rollouts, elapsed_time))
            
        loop.close()
        
        # Print results
        print("\nTime limit comparison:")
        for time_limit, depth, rollouts, elapsed_time in results:
            print(f"Time limit: {time_limit}s, depth={depth}, rollouts={rollouts}, actual time: {elapsed_time:.2f}s")
            
        # Verify that longer time limits result in more rollouts
        for i in range(1, len(results)):
            self.assertTrue(
                results[i][2] >= results[i-1][2], 
                f"More time should result in at least as many rollouts: {results[i][0]}s vs {results[i-1][0]}s"
            )
            
    def test_get_action_with_different_epochs(self):
        """Test get_action with different epoch values"""
        epoch_values = [1, 5, 10]
        results = []
        
        # Create event loop for async functions
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        for epochs in epoch_values:
            start_time = time.time()
            action, depth, rollouts = loop.run_until_complete(
                gnn.montecarlo_alg.get_action(
                    self.model, 
                    max_time_for_dec=3,
                    epochs=epochs,
                    X_train=self.x_train,
                    Y_train=self.y_train,
                    simulation_score=self.simulation_score
                )
            )
            elapsed_time = time.time() - start_time
            
            results.append((epochs, depth, rollouts, elapsed_time))
            
        loop.close()
        
        # Print results
        print("\nEpochs comparison:")
        for epochs, depth, rollouts, elapsed_time in results:
            print(f"Epochs: {epochs}, depth={depth}, rollouts={rollouts}, time={elapsed_time:.2f}s")
            
        # Verify that more epochs result in fewer rollouts (due to time constraint)
        for i in range(1, len(results)):
            self.assertTrue(
                results[i][2] <= results[i-1][2], 
                f"More epochs should result in fewer rollouts due to time constraint: {results[i][0]} vs {results[i-1][0]}"
            )

if __name__ == '__main__':
    unittest.main() 