import unittest
import numpy as np
import sys
sys.path.append('.')
sys.path.append('../')
import growingnn as gnn
from growingnn.helpers import one_hot, get_numpy_array, convert_to_desired_type, train_test_split_many_inputs, protected_sampling, select_data_at_indices, limit_classes

class TestDataPreprocessing(unittest.TestCase):
    def setUp(self):
        # Test data for one-hot encoding
        self.y_labels = np.array([0, 1, 2, 0, 1])
        
        # Test data for train-test split
        self.X = np.random.rand(10, 5)  # 10 samples, 5 features
        self.y = np.random.randint(0, 3, size=10)  # 3 classes
        
        # Test data for protected sampling
        self.X_protected = np.random.rand(20, 5)  # 20 samples, 5 features
        self.y_protected = np.array([0]*5 + [1]*5 + [2]*5 + [3]*5)  # 4 classes, 5 samples each
        
    def test_one_hot_encoding(self):
        # Test basic one-hot encoding
        y_one_hot = one_hot(self.y_labels)
        expected = np.array([
            [1, 0, 0, 1, 0],  # class 0
            [0, 1, 0, 0, 1],  # class 1
            [0, 0, 1, 0, 0]   # class 2
        ])
        np.testing.assert_array_equal(y_one_hot, expected)
        
        # Test with custom Y_max
        y_one_hot = one_hot(self.y_labels, Y_max=4)
        expected = np.array([
            [1, 0, 0, 1, 0],  # class 0
            [0, 1, 0, 0, 1],  # class 1
            [0, 0, 1, 0, 0],  # class 2
            [0, 0, 0, 0, 0]   # class 3
        ])
        np.testing.assert_array_equal(y_one_hot, expected)
        
    def test_get_numpy_array(self):
        # Test with numpy array
        x = np.array([1, 2, 3])
        result = get_numpy_array(x)
        np.testing.assert_array_equal(result, x)
        
        # Test with list
        x = [1, 2, 3]
        result = get_numpy_array(x)
        np.testing.assert_array_equal(result, np.array(x))
        
        # Test with nested lists
        x = [[1, 2], [3, 4]]
        result = get_numpy_array(x)
        np.testing.assert_array_equal(result, np.array(x))
        
    def test_convert_to_desired_type(self):
        # Test with numpy array
        x = np.array([1, 2, 3])
        result = convert_to_desired_type(x)
        np.testing.assert_array_equal(result, x)
        
        # Test with list
        x = [1, 2, 3]
        result = convert_to_desired_type(x)
        np.testing.assert_array_equal(result, np.array(x))
        
        # Test with scalar
        x = 5
        result = convert_to_desired_type(x)
        self.assertEqual(result, 5)
        
    def test_train_test_split(self):
        # Test with different test sizes
        test_sizes = [0.1, 0.2, 0.3, 0.4]
        
        for test_size in test_sizes:
            X_train, X_test, y_train, y_test = train_test_split_many_inputs(self.X, self.y, test_size)
            
            # Check shapes
            expected_train_size = int(self.X.shape[1] * (1 - test_size))
            expected_test_size = int(self.X.shape[1] * test_size)
            
            self.assertEqual(X_train.shape[1], expected_train_size)
            self.assertEqual(X_test.shape[1], expected_test_size)
            self.assertEqual(y_train.shape[0], expected_train_size)
            self.assertEqual(y_test.shape[0], expected_test_size)
            
            # Check if all classes are represented in both sets
            train_classes = np.unique(y_train)
            test_classes = np.unique(y_test)
            all_classes = np.unique(self.y)
            
            self.assertTrue(np.all(np.isin(all_classes, train_classes)))
            self.assertTrue(np.all(np.isin(all_classes, test_classes)))
            
    def test_protected_sampling(self):
        # Test with different sample sizes
        sample_sizes = [4, 8, 12, 16]
        
        for n in sample_sizes:
            X_sampled, y_sampled = protected_sampling(self.X_protected, self.y_protected, n)
            
            # Check shapes
            self.assertEqual(X_sampled.shape[1], n)
            self.assertEqual(y_sampled.shape[0], n)
            
            # Check if all classes are represented
            unique_classes = np.unique(y_sampled)
            self.assertEqual(len(unique_classes), min(n, len(np.unique(self.y_protected))))
            
            # Check if samples are balanced
            class_counts = np.bincount(y_sampled)
            max_count = np.max(class_counts)
            min_count = np.min(class_counts)
            self.assertTrue(max_count - min_count <= 1)  # Allow for 1 sample difference
            
    def test_select_data_at_indices(self):
        # Test with different index sets
        indices_sets = [
            [0, 2, 4],  # First half
            [1, 3, 5],  # Second half
            [0, 1, 2],  # First three
            [7, 8, 9]   # Last three
        ]
        
        for indices in indices_sets:
            X_selected, y_selected = select_data_at_indices(self.X, self.y, indices)
            
            # Check shapes
            self.assertEqual(X_selected.shape[1], len(indices))
            self.assertEqual(y_selected.shape[0], len(indices))
            
            # Check if correct samples were selected
            np.testing.assert_array_equal(X_selected, self.X[:, indices])
            np.testing.assert_array_equal(y_selected, self.y[indices])
            
    def test_limit_classes(self):
        # Test with different class limits
        num_classes_list = [1, 2, 3]
        
        for num_classes in num_classes_list:
            X_train_limited, y_train_limited, X_test_limited, y_test_limited = limit_classes(
                self.X, self.y, self.X, self.y, num_classes
            )
            
            # Check if classes are limited
            self.assertTrue(np.all(y_train_limited < num_classes))
            self.assertTrue(np.all(y_test_limited < num_classes))
            
            # Check if shapes are preserved
            self.assertEqual(X_train_limited.shape, self.X.shape)
            self.assertEqual(X_test_limited.shape, self.X.shape)
            
    def test_edge_cases(self):
        # Test with empty arrays
        empty_X = np.array([])
        empty_y = np.array([])
        
        with self.assertRaises(ValueError):
            train_test_split_many_inputs(empty_X, empty_y, 0.2)
            
        with self.assertRaises(ValueError):
            protected_sampling(empty_X, empty_y, 5)
            
        with self.assertRaises(ValueError):
            select_data_at_indices(empty_X, empty_y, [0, 1])
            
        # Test with invalid indices
        with self.assertRaises(IndexError):
            select_data_at_indices(self.X, self.y, [100])  # Out of bounds
            
        # Test with invalid test size
        with self.assertRaises(ValueError):
            train_test_split_many_inputs(self.X, self.y, 1.5)  # > 1.0
            
        with self.assertRaises(ValueError):
            train_test_split_many_inputs(self.X, self.y, -0.1)  # < 0.0
            
    def test_numerical_stability(self):
        # Test with very large values
        large_X = np.array([[1e10, 2e10], [3e10, 4e10]])
        large_y = np.array([0, 1])
        
        # All functions should handle large values
        X_train, X_test, y_train, y_test = train_test_split_many_inputs(large_X, large_y, 0.5)
        self.assertTrue(np.all(np.isfinite(X_train)))
        self.assertTrue(np.all(np.isfinite(X_test)))
        
        X_sampled, y_sampled = protected_sampling(large_X, large_y, 1)
        self.assertTrue(np.all(np.isfinite(X_sampled)))
        
        X_selected, y_selected = select_data_at_indices(large_X, large_y, [0])
        self.assertTrue(np.all(np.isfinite(X_selected)))

if __name__ == '__main__':
    unittest.main() 