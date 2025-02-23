import unittest

from tensordict import TensorDict
from rlarcworld.utils import TorchQueue

import unittest
import torch


class TestTorchQueue(unittest.TestCase):
    def test_init(self):
        queue = TorchQueue(torch.randn(10), 10)
        self.assertEqual(queue._queue_size, 10)
        self.assertEqual(queue._queue_dim, 0)
        self.assertRaises(AssertionError, TorchQueue, torch.randn(10), -1)
        self.assertRaises(AssertionError, TorchQueue, torch.randn(10), (0,))
        self.assertRaises(AssertionError, TorchQueue, torch.randn(10), 10, (0,))
        queue = queue.clone()
        queue = TensorDict({"Hello": queue}).to("cuda")

    def test_push(self):
        queue = TorchQueue(torch.randn(5), 10)
        item = torch.randn(1)
        queue = queue.push(item)
        self.assertEqual(queue.shape[queue.queue_dim], 6)
        self.assertEqual(queue[-1], item)

    def test_push_multiple_dimensions(self):
        queue = TorchQueue(torch.randn(3, 5), 10, queue_dim=1)
        item = torch.randn(3, 1)
        queue = queue.push(item)
        self.assertEqual(queue.shape[queue.queue_dim], 6)
        torch.testing.assert_close(queue[:, -1], item.squeeze(1))

    def test_pop(self):
        item = torch.randn(10)
        queue = TorchQueue(item, 10)
        poped, queue = queue.pop()
        self.assertEqual(queue.shape[queue.queue_dim], 9)
        self.assertEqual(queue[-1], item[-1])
        self.assertEqual(poped, item[0])

    def test_multiple_push(self):
        queue = TorchQueue(torch.randn(5), 10)
        for i in range(15):
            item = torch.randn(1)
            queue = queue.push(item)
            self.assertEqual(queue.shape[queue._queue_dim], min(5 + i + 1, 10))
            self.assertEqual(queue[-1], item)

    def test_multiple_push_multiple_dimensions(self):
        queue = TorchQueue(torch.randn(3, 5), 10, queue_dim=1)
        for i in range(15):
            item = torch.randn(3, 1)
            queue = queue.push(item)
            self.assertEqual(queue.shape[queue._queue_dim], min(5 + i + 1, 10))
            torch.testing.assert_close(queue[:, -1], item.squeeze(1))

    def test_multiple_pop(self):
        queue = TorchQueue(torch.randn(10), 10)
        last_value = queue[-1]
        for i in range(5):
            oldest_value = queue[0]
            item, queue = queue.pop()
            self.assertEqual(queue.shape[queue._queue_dim], 10 - i - 1)
            self.assertEqual(last_value, queue[-1])
            self.assertEqual(oldest_value, item)

    def test_queue_full(self):
        queue = TorchQueue(torch.randn(10), 10)
        item = torch.randn(1)
        queue = queue.push(item)
        self.assertEqual(queue.shape[queue._queue_dim], 10)
        self.assertEqual(queue[-1], item)

    def test_queue_empty(self):
        queue = TorchQueue(torch.randn(0), 10)
        with self.assertRaises(IndexError):
            queue.pop()
        self.assertEqual(queue.shape[queue._queue_dim], 0)


if __name__ == "__main__":
    unittest.main()
