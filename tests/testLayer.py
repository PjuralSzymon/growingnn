import sys
sys.path.append('.')
sys.path.append('../')
import growingnn as gnn
import unittest
from testSuite import mode
from unittest.mock import MagicMock

class TestLayer(unittest.TestCase):

    def setUp(self):
        global mode
        if mode == 'cpu':
            gnn.switch_to_cpu()
        elif mode == 'gpu':
            gnn.switch_to_gpu()
        self.layer = gnn.Layer(1, None, 10, 5, None)
        self.layer.act_fun = gnn.structure.Activations.ReLu
        self.layer.set_as_ending()

    def test_init(self):
        self.assertEqual(self.layer.id, 1)
        self.assertIsNone(self.layer.model)
        self.assertEqual(self.layer.input_size, 10)
        self.assertEqual(self.layer.neurons, 5)
        self.assertEqual(self.layer.input_layers_ids, [])
        self.assertEqual(self.layer.output_layers_ids, [])
        self.assertEqual(self.layer.f_input, [])
        self.assertEqual(self.layer.b_input, [])
        self.assertEqual(self.layer.reshspers, {})

    def test_connect_input(self):
        self.layer.model = MagicMock()
        self.layer.model.get_layer.return_value = MagicMock(output_layers_ids=[])
        self.layer.connect_input(2)
        self.assertIn(2, self.layer.input_layers_ids)
        self.layer.model.get_layer.assert_called_with(2)
        self.layer.model.get_layer.return_value.connect_output.assert_called_with(1)

    def test_connect_output(self):
        self.layer.model = MagicMock()
        self.layer.model.get_layer.return_value = MagicMock(input_layers_ids=[])
        self.layer.connect_output(2)
        self.assertIn(2, self.layer.output_layers_ids)
        self.layer.model.get_layer.assert_called_with(2)
        self.layer.model.get_layer.return_value.connect_input.assert_called_with(1)

    def test_disconnect(self):
        self.layer.input_layers_ids = [2, 3]
        self.layer.output_layers_ids = [4, 5]
        self.layer.disconnect(3)
        self.assertNotIn(3, self.layer.input_layers_ids)
        self.assertEqual(self.layer.output_layers_ids, [4, 5])
        self.layer.disconnect(5)
        self.assertEqual(self.layer.output_layers_ids, [4])

    def test_forward_prop(self):
        self.layer.input_layers_ids = [2, 3]
        self.layer.model = MagicMock()
        self.layer.forward_prop(gnn.np.ones((10, 1)))
        self.layer.forward_prop(gnn.np.ones((10, 1)))
        result2 = self.layer.A
        self.assertIsNotNone(result2)
        self.assertEqual(len(self.layer.f_input), 0)

    def test_back_prop(self):
        self.layer.output_layers_ids = [2, 3]
        self.layer.model = MagicMock()

        self.layer.E = gnn.np.ones((5, 1))
        self.layer.b_input = [gnn.np.ones((5, 1)), gnn.np.ones((5, 1))]

        self.layer.forward_prop(gnn.np.ones((10, 1)))
        self.layer.back_prop(gnn.np.ones((5, 1)), 1, 0.01)

        self.assertEqual(len(self.layer.b_input), 0)
        self.assertTrue(hasattr(self.layer, 'dW'))
        self.assertTrue(hasattr(self.layer, 'dB'))

if __name__ == '__main__':
    unittest.main()
