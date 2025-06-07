import sys
sys.path.append('.')
sys.path.append('../')
import growingnn as gnn
import numpy as np
import unittest
from testSuite import mode

class TestHelpers(unittest.TestCase):
    def setUp(self):
        global mode
        if mode == 'cpu':
            gnn.switch_to_cpu()
        elif mode == 'gpu':
            gnn.switch_to_gpu()

    def test_one_hot(self):
        # Test basic one-hot encoding
        Y = np.array([0, 1, 2, 0])
        Y_one_hot = gnn.helpers.one_hot(Y)
        expected = np.array([
            [1, 0, 0, 1],  # class 0
            [0, 1, 0, 0],  # class 1
            [0, 0, 1, 0]   # class 2
        ])
        np.testing.assert_array_equal(Y_one_hot, expected)

        # Test with custom Y_max
        Y_one_hot = gnn.helpers.one_hot(Y, Y_max=4)
        expected = np.array([
            [1, 0, 0, 1],  # class 0
            [0, 1, 0, 0],  # class 1
            [0, 0, 1, 0],  # class 2
            [0, 0, 0, 0]   # class 3
        ])
        np.testing.assert_array_equal(Y_one_hot, expected)

    def test_get_numpy_array(self):
        # Test with numpy array
        x = np.array([1, 2, 3])
        result = gnn.helpers.get_numpy_array(x)
        np.testing.assert_array_equal(result, x)

        # Test with list
        x = [1, 2, 3]
        result = gnn.helpers.get_numpy_array(x)
        np.testing.assert_array_equal(result, np.array(x))

    def test_convert_to_desired_type(self):
        # Test with numpy array
        x = np.array([1, 2, 3])
        result = gnn.helpers.convert_to_desired_type(x)
        np.testing.assert_array_equal(result, x)

        # Test with list
        x = [1, 2, 3]
        result = gnn.helpers.convert_to_desired_type(x)
        np.testing.assert_array_equal(result, np.array(x))

    def test_clip(self):
        # Test basic clipping
        x = np.array([-1, 0, 1, 2, 3])
        result = gnn.helpers.clip(x, 0, 2)
        expected = np.array([0, 0, 1, 2, 2])
        np.testing.assert_array_equal(result, expected)

    def test_argmax(self):
        # Test basic argmax
        x = np.array([
            [1, 2, 3],
            [4, 5, 6]
        ])
        result = gnn.helpers.argmax(x, axis=0)
        expected = np.array([1, 1, 1])
        np.testing.assert_array_equal(result, expected)

        result = gnn.helpers.argmax(x, axis=1)
        expected = np.array([2, 2])
        np.testing.assert_array_equal(result, expected)

    def test_get_reverse_normal_distribution(self):
        # Test shape and range
        shape = (5, 5)
        clip_range = 1.0
        
        # Calculate total number of samples needed
        num_samples = np.prod(shape)
        
        # Generate distribution
        result = gnn.helpers.get_reverse_normal_distribution(clip_range, shape)
        
        # Check shape
        self.assertEqual(result.shape, shape)
        
        # Check range (should be roughly within -2.2*clip_range to 2.2*clip_range)
        self.assertTrue(np.all(result >= -2.2 * clip_range))
        self.assertTrue(np.all(result <= 2.2 * clip_range))
        
        # Check that we have enough samples
        self.assertEqual(result.size, num_samples)

    def test_train_test_split_many_inputs(self):
        # Test with 2D input
        x = np.random.rand(10, 5)  # (features, samples)
        y = np.random.randint(3, size=(5,))  # (samples,)
        test_size = 0.2
        
        x_train, x_test, y_train, y_test = gnn.helpers.train_test_split_many_inputs(x, y, test_size)
        
        # Check shapes
        self.assertEqual(x_train.shape[1], 4)  # 80% of 5 samples
        self.assertEqual(x_test.shape[1], 1)   # 20% of 5 samples
        self.assertEqual(y_train.shape[0], 4)
        self.assertEqual(y_test.shape[0], 1)

    def test_protected_sampling(self):
        # Test with balanced classes
        x = np.random.rand(10, 6)  # (features, samples)
        y = np.array([0, 0, 1, 1, 2, 2])  # 2 samples per class
        n = 3  # Request 3 samples
        
        x_sampled, y_sampled = gnn.helpers.protected_sampling(x, y, n)
        
        # Check that we have at least one sample from each class
        unique_classes = np.unique(y_sampled)
        self.assertEqual(len(unique_classes), 3)

    def test_select_data_at_indices(self):
        # Test with 2D input
        x = np.random.rand(10, 5)  # (features, samples)
        y = np.random.randint(3, size=(5,))  # (samples,)
        indices = [0, 2, 4]
        
        x_selected, y_selected = gnn.helpers.select_data_at_indices(x, y, indices)
        
        # Check shapes
        self.assertEqual(x_selected.shape[1], len(indices))
        self.assertEqual(y_selected.shape[0], len(indices))

    def test_limit_classes(self):
        # Test limiting number of classes
        x_train = np.random.rand(10, 5)
        y_train = np.array([0, 1, 2, 3, 4])
        x_test = np.random.rand(10, 5)
        y_test = np.array([0, 1, 2, 3, 4])
        num_classes = 3
        
        x_train_limited, y_train_limited, x_test_limited, y_test_limited = gnn.helpers.limit_classes(
            x_train, y_train, x_test, y_test, num_classes
        )
        
        # Check that only classes 0, 1, 2 are present
        self.assertTrue(np.all(y_train_limited < num_classes))
        self.assertTrue(np.all(y_test_limited < num_classes))

if __name__ == '__main__':
    unittest.main() 