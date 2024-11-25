import unittest
import asyncio
import sys
sys.path.append('.')
sys.path.append('../')
import growingnn as gnn
import numpy as np

# Import your script here, e.g., 
# from myscript import get_reshsper, Reshape, Reshape_forward_prop, Reshape_back_prop, eye_stretch

MAX_ENTRIES = 100  # Match with the script configuration


class TestReshapeFunctions(unittest.TestCase):

    def test_eye_stretch(self):
        a, b = 5, 10
        result = gnn.quaziIdentity.eye_stretch(a, b)
        self.assertEqual(result.shape, (a, b))

    def test_get_reshsper(self):
        size_from, size_to = 10, 20

        # First time, it should create a new resherper
        resherper = gnn.quaziIdentity.get_reshsper(size_from, size_to)
        self.assertIsNotNone(resherper)
        self.assertEqual(resherper.shape, (size_from, size_to))

        # Second time, it should fetch the same one
        resherper_again = gnn.quaziIdentity.get_reshsper(size_from, size_to)
        self.assertTrue(np.array_equal(resherper, resherper_again))

        # Test identity scenario (no reshape needed)
        self.assertIsNone(gnn.quaziIdentity.get_reshsper(10, 10))

    def test_reshape(self):
        input_data = np.random.rand(10, 5)  # 10 features, 5 samples
        QIdentity = gnn.quaziIdentity.eye_stretch(10, 20)
        reshaped_data = gnn.quaziIdentity.Reshape(input_data, 20, QIdentity)

        self.assertEqual(reshaped_data.shape, (20, 5))
        self.assertTrue(np.allclose(np.dot(input_data[:, 0], QIdentity), reshaped_data[:, 0]))

    def test_reshape_forward_prop(self):
        input_data = np.random.rand(5, 3, 3, 2)  # Batch of 5, 3x3x2 feature maps
        flatten_size = 3 * 3 * 2
        QIdentity = gnn.quaziIdentity.eye_stretch(flatten_size, 10)
        reshaped_data = gnn.quaziIdentity.Reshape_forward_prop(input_data, 10, QIdentity)

        self.assertEqual(reshaped_data.shape, (10, 5))

    def test_reshape_back_prop(self):
        output_size, input_shape = 10, (3, 3, 2)
        E = np.random.rand(10, 5)
        QIdentity = gnn.quaziIdentity.eye_stretch(output_size, np.prod(input_shape))
        back_prop_data = gnn.quaziIdentity.Reshape_back_prop(E, input_shape, QIdentity)

        self.assertEqual(back_prop_data.shape, (5, 3, 3, 2))


if __name__ == '__main__':
    unittest.main()
