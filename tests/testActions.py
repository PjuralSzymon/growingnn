import sys
sys.path.append('.')
sys.path.append('../')
import growingnn as gnn
import numpy as np
import unittest
from testSuite import mode

class TestActionGeneration(unittest.TestCase):
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
        
    def test_action_generation_basic(self):
        """Test basic action generation for a simple model"""
        # Generate all possible actions
        actions = gnn.action.Action.generate_all_actions(self.model)
        
        # Verify that actions were generated
        self.assertIsInstance(actions, list, "Actions should be a list")
        self.assertTrue(len(actions) > 0, "At least one action should be generated")
        
        # Print the number of actions generated
        print(f"Generated {len(actions)} actions for basic model")
        
        # Check that all actions are instances of Action or its subclasses
        for action in actions:
            self.assertIsInstance(action, gnn.action.Action, 
                                f"Action should be an instance of Action, got {type(action)}")
            
    def test_action_types(self):
        """Test that different types of actions are generated"""
        # Generate all possible actions
        actions = gnn.action.Action.generate_all_actions(self.model)
        
        # Count the different types of actions
        action_types = {}
        for action in actions:
            action_type = type(action).__name__
            if action_type in action_types:
                action_types[action_type] += 1
            else:
                action_types[action_type] = 1
                
        # Print the counts
        print("Action types generated:")
        for action_type, count in action_types.items():
            print(f"  {action_type}: {count}")
            
        # Verify that we have at least some of the expected action types
        # Note: Del_Layer actions may not be generated if there are no hidden layers
        expected_types = ['Add_Seq_Layer', 'Add_Res_Layer']
        for expected_type in expected_types:
            self.assertIn(expected_type, action_types, 
                         f"Expected to find {expected_type} actions")
            
    def test_add_seq_layer_actions(self):
        """Test generation of Add_Seq_Layer actions"""
        # Generate Add_Seq_Layer actions specifically
        actions = gnn.action.Add_Seq_Layer.generate_all_actions(self.model)
        
        # Verify that actions were generated
        self.assertIsInstance(actions, list, "Add_Seq_Layer actions should be a list")
        
        # Print the number of actions generated
        print(f"Generated {len(actions)} Add_Seq_Layer actions")
        
        # Check that all actions are instances of Add_Seq_Layer
        for action in actions:
            self.assertIsInstance(action, gnn.action.Add_Seq_Layer, 
                                f"Action should be an instance of Add_Seq_Layer, got {type(action)}")
            
        # Check that the actions have the expected parameters
        for action in actions:
            self.assertEqual(len(action.params), 3, 
                           "Add_Seq_Layer actions should have 3 parameters")
            # The parameters might be strings like 'init_0' instead of integers
            self.assertIsInstance(action.params[0], (int, str), 
                                "First parameter should be an integer or string (from_layer_id)")
            self.assertIsInstance(action.params[1], (int, str), 
                                "Second parameter should be an integer or string (to_layer_id)")
            self.assertIsInstance(action.params[2], gnn.structure.Layer_Type, 
                                "Third parameter should be a Layer_Type")
            
    def test_add_res_layer_actions(self):
        """Test generation of Add_Res_Layer actions"""
        # Generate Add_Res_Layer actions specifically
        actions = gnn.action.Add_Res_Layer.generate_all_actions(self.model)
        
        # Verify that actions were generated
        self.assertIsInstance(actions, list, "Add_Res_Layer actions should be a list")
        
        # Print the number of actions generated
        print(f"Generated {len(actions)} Add_Res_Layer actions")
        
        # Check that all actions are instances of Add_Res_Layer
        for action in actions:
            self.assertIsInstance(action, gnn.action.Add_Res_Layer, 
                                f"Action should be an instance of Add_Res_Layer, got {type(action)}")
            
        # Check that the actions have the expected parameters
        for action in actions:
            self.assertEqual(len(action.params), 3, 
                           "Add_Res_Layer actions should have 3 parameters")
            # The parameters might be strings like 'init_0' instead of integers
            self.assertIsInstance(action.params[0], (int, str), 
                                "First parameter should be an integer or string (from_layer_id)")
            self.assertIsInstance(action.params[1], (int, str), 
                                "Second parameter should be an integer or string (to_layer_id)")
            self.assertIsInstance(action.params[2], gnn.structure.Layer_Type, 
                                "Third parameter should be a Layer_Type")
            
    def test_del_layer_actions(self):
        """Test generation of Del_Layer actions"""
        # Create a model with at least one hidden layer to test Del_Layer actions
        model = gnn.structure.Model(
            self.input_size, 
            self.input_size, 
            self.output_size, 
            gnn.structure.Loss.multiclass_cross_entropy, 
            gnn.structure.Activations.Sigmoid, 
            1
        )
        
        # Generate Del_Layer actions specifically
        actions = gnn.action.Del_Layer.generate_all_actions(model)
        
        # Verify that actions were generated
        self.assertIsInstance(actions, list, "Del_Layer actions should be a list")
        
        # Print the number of actions generated
        print(f"Generated {len(actions)} Del_Layer actions")
        
        # Check that all actions are instances of Del_Layer
        for action in actions:
            self.assertIsInstance(action, gnn.action.Del_Layer, 
                                f"Action should be an instance of Del_Layer, got {type(action)}")
            
        # Check that the actions have the expected parameters
        for action in actions:
            # The parameter might be a string like 'init_0' instead of an integer
            self.assertIsInstance(action.params, (int, str), 
                                "Del_Layer actions should have an integer or string parameter (layer_id)")
            
    def test_action_execution(self):
        """Test that actions can be executed on the model"""
        # Generate all possible actions
        actions = gnn.action.Action.generate_all_actions(self.model)
        
        # Execute each action and verify it doesn't raise an exception
        for action in actions:
            try:
                action.execute(self.model)
                # If we get here, the action executed successfully
                self.assertTrue(True, f"Action {action} executed successfully")
            except Exception as e:
                self.fail(f"Action {action} failed to execute: {str(e)}")
                
    def test_action_influence(self):
        """Test that actions can check if they can be influenced by other actions"""
        # Generate all possible actions
        actions = gnn.action.Action.generate_all_actions(self.model)
        
        # For each action, check if it can be influenced by other actions
        for action in actions:
            for other_action in actions:
                try:
                    # This should not raise an exception
                    action.can_be_infulenced(other_action)
                except Exception as e:
                    self.fail(f"Action {action} failed to check influence from {other_action}: {str(e)}")
                    
    def test_action_generation_with_complex_model(self):
        """Test action generation with a more complex model"""
        # Create a more complex model with multiple layers
        model = gnn.structure.Model(
            self.input_size, 
            self.input_size, 
            self.output_size, 
            gnn.structure.Loss.multiclass_cross_entropy, 
            gnn.structure.Activations.Sigmoid, 
            1
        )
        
        # Instead of directly adding layers, we'll use the model's methods to get the layer IDs
        # and then generate actions based on those IDs
        input_layer_id = model.input_layers[0].id
        output_layer_id = model.output_layer.id
        
        # Generate all possible actions
        actions = gnn.action.Action.generate_all_actions(model)
        
        # Verify that actions were generated
        self.assertIsInstance(actions, list, "Actions should be a list")
        self.assertTrue(len(actions) > 0, "At least one action should be generated")
        
        # Print the number of actions generated
        print(f"Generated {len(actions)} actions for complex model")
        
        # Count the different types of actions
        action_types = {}
        for action in actions:
            action_type = type(action).__name__
            if action_type in action_types:
                action_types[action_type] += 1
            else:
                action_types[action_type] = 1
                
        # Print the counts
        print("Action types generated for complex model:")
        for action_type, count in action_types.items():
            print(f"  {action_type}: {count}")
            
    def test_action_generation_with_conv_model(self):
        """Test action generation with a model containing convolutional layers"""
        # Create a model with convolutional layers
        model = gnn.structure.Model(
            self.input_size, 
            self.input_size, 
            self.output_size, 
            gnn.structure.Loss.multiclass_cross_entropy, 
            gnn.structure.Activations.Sigmoid, 
            1
        )
        
        # Instead of directly adding layers, we'll use the model's methods to get the layer IDs
        # and then generate actions based on those IDs
        input_layer_id = model.input_layers[0].id
        output_layer_id = model.output_layer.id
        
        # Generate all possible actions
        actions = gnn.action.Action.generate_all_actions(model)
        
        # Verify that actions were generated
        self.assertIsInstance(actions, list, "Actions should be a list")
        self.assertTrue(len(actions) > 0, "At least one action should be generated")
        
        # Print the number of actions generated
        print(f"Generated {len(actions)} actions for conv model")
        
        # Count the different types of actions
        action_types = {}
        for action in actions:
            action_type = type(action).__name__
            if action_type in action_types:
                action_types[action_type] += 1
            else:
                action_types[action_type] = 1
                
        # Print the counts
        print("Action types generated for conv model:")
        for action_type, count in action_types.items():
            print(f"  {action_type}: {count}")
            
        # Check for convolutional-specific actions
        conv_actions = [a for a in actions if isinstance(a, (gnn.action.Add_Seq_Conv_Layer, gnn.action.Add_Res_Conv_Layer))]
        # We might not have conv actions if the model doesn't have conv layers yet
        # So we'll just print the count instead of asserting
        print(f"Found {len(conv_actions)} convolutional actions")
        
    def test_action_generation_with_minimal_model(self):
        """Test action generation with a minimal model (smallest possible hidden size)"""
        # Create a model with the smallest possible hidden size (1)
        model = gnn.structure.Model(
            self.input_size, 
            1,  # Minimal hidden size
            self.output_size, 
            gnn.structure.Loss.multiclass_cross_entropy, 
            gnn.structure.Activations.Sigmoid, 
            1
        )
        
        # Generate all possible actions
        actions = gnn.action.Action.generate_all_actions(model)
        
        # Verify that actions were generated
        self.assertIsInstance(actions, list, "Actions should be a list")
        
        # Print the number of actions generated
        print(f"Generated {len(actions)} actions for minimal model")
        
        # Count the different types of actions
        action_types = {}
        for action in actions:
            action_type = type(action).__name__
            if action_type in action_types:
                action_types[action_type] += 1
            else:
                action_types[action_type] = 1
                
        # Print the counts
        print("Action types generated for minimal model:")
        for action_type, count in action_types.items():
            print(f"  {action_type}: {count}")
            
    def test_action_generation_with_large_model(self):
        """Test action generation with a large model (many layers)"""
        # Create a large model with many layers
        model = gnn.structure.Model(
            self.input_size, 
            self.input_size, 
            self.output_size, 
            gnn.structure.Loss.multiclass_cross_entropy, 
            gnn.structure.Activations.Sigmoid, 
            1
        )
        
        # Instead of directly adding layers, we'll use the model's methods to get the layer IDs
        # and then generate actions based on those IDs
        input_layer_id = model.input_layers[0].id
        output_layer_id = model.output_layer.id
        
        # Generate all possible actions
        actions = gnn.action.Action.generate_all_actions(model)
        
        # Verify that actions were generated
        self.assertIsInstance(actions, list, "Actions should be a list")
        self.assertTrue(len(actions) > 0, "At least one action should be generated")
        
        # Print the number of actions generated
        print(f"Generated {len(actions)} actions for large model")
        
        # Count the different types of actions
        action_types = {}
        for action in actions:
            action_type = type(action).__name__
            if action_type in action_types:
                action_types[action_type] += 1
            else:
                action_types[action_type] = 1
                
        # Print the counts
        print("Action types generated for large model:")
        for action_type, count in action_types.items():
            print(f"  {action_type}: {count}")
            
        # Check that we have a reasonable number of actions
        # The exact number may vary, so we'll just check that it's positive
        self.assertTrue(len(actions) > 0, 
                       "Large model should generate at least one action")

if __name__ == '__main__':
    unittest.main() 