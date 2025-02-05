import sys
sys.path.append('.')
sys.path.append('../')
import growingnn as gnn
import unittest
from testSuite import mode

class TestHistory(unittest.TestCase):
    def setUp(self):
        global mode
        if mode == 'cpu':
            gnn.switch_to_cpu()
        elif mode == 'gpu':
            gnn.switch_to_gpu()
        self.keys = ['loss', 'accuracy']
        self.history = gnn.History(self.keys)

    def test_init(self):
        self.assertDictEqual(self.history.Y, {'loss': [], 'accuracy': []})
        self.assertEqual(self.history.last_img_id, 0)
        self.assertEqual(self.history.description, "\ndescription_of_training_process: \n")
        self.assertEqual(self.history.best_train_acc, 0.0)
        self.assertEqual(self.history.best_test_acc, 0.0)

    def test_get_length(self):
        self.assertEqual(self.history.get_length(), 0)
        self.history.Y['loss'] = [0.3, 0.4, 0.5]
        self.assertEqual(self.history.get_length(), 3)

    def test_append(self):
        self.history.append('loss', 0.3)
        self.history.append('accuracy', 0.9)
        self.assertEqual(self.history.Y['loss'], [0.3])
        self.assertEqual(self.history.Y['accuracy'], [0.9])

    def test_merge(self):
        new_hist = gnn.History(self.keys)
        new_hist.append('loss', 0.4)
        new_hist.append('accuracy', 0.8)
        self.history.Y['loss'] = [0.3]
        self.history.Y['accuracy'] = [0.9]
        self.history.merge(new_hist)
        self.assertEqual(self.history.Y['loss'], [0.3, 0.4])
        self.assertEqual(self.history.Y['accuracy'], [0.9, 0.8])

    def test_learning_capable_below_threshold(self):
        self.history.Y['loss'] = [0.1] * 9
        self.assertTrue(self.history.learning_capable())
        
    def test_get_last(self):
        self.history.Y['loss'] = [0.3, 0.4, 0.5]
        self.assertEqual(self.history.get_last('loss'), 0.5)


if __name__ == '__main__':
    unittest.main()
