import asyncio
import sys
sys.path.append('.')
sys.path.append('../')
import growingnn as gnn
import unittest
import pandas as pd
import random
from unittest.mock import MagicMock

class TestModel(unittest.TestCase):

    def setUp(self):
        self.model = gnn.Model(input_size=10, hidden_size=5, output_size=3)

    def test_init(self):
        self.assertEqual(self.model.batch_size, 128)
        self.assertEqual(self.model.input_size, 10)
        self.assertEqual(self.model.output_size, 3)
        self.assertEqual(self.model.hidden_size, 5)
        self.assertEqual(len(self.model.hidden_layers), 0)
        self.assertEqual(self.model.avaible_id, 2)
        self.assertEqual(self.model.convolution, False)
        self.assertIsNone(self.model.input_shape)
        self.assertIsNone(self.model.kernel_size)
        self.assertIsNone(self.model.depth)

    def test_forward_prop(self):
        input_data = gnn.np.random.rand(10, 10)  # Example input data
        result = self.model.forward_prop(input_data)
        self.assertEqual(result.shape, (3, 10))  # Assuming output_size is 3

    def test_add_and_remove_layer(self):
        self.assertEqual(len(self.model.hidden_layers), 0)
        layer_id = self.model.add_res_layer(layer_from_id='init_0', layer_to_id=1)
        self.assertEqual(len(self.model.hidden_layers), 1)
        self.model.remove_layer(layer_id)
        self.assertEqual(len(self.model.hidden_layers), 0)

if __name__ == '__main__':
    unittest.main()
