import pytest
import torch
from tensordict import TensorDict

from rlarcworld.utils.torch_queue import TorchQueue


def test_torch_queue_init():
    queue = TorchQueue(torch.randn(10), 10)
    assert queue._queue_size == 10
    assert queue._queue_dim == 0
    with pytest.raises(AssertionError):
        TorchQueue(torch.randn(10), -1)
    with pytest.raises(AssertionError):
        TorchQueue(torch.randn(10), (0,))
    with pytest.raises(AssertionError):
        TorchQueue(torch.randn(10), 10, (0,))

    # clone() should work; TensorDict should be able to carry it
    queue = queue.clone()
    td = TensorDict({"Hello": queue}, batch_size=[])
    # Moving to CUDA may fail on CPU-only machines;
    # ensure it doesn't crash the test suite
    if torch.cuda.is_available():
        td_cuda = td.to("cuda")
        assert isinstance(td_cuda, TensorDict)


def test_torch_queue_push():
    queue = TorchQueue(torch.randn(5), 10)
    item = torch.randn(1)
    queue = queue.push(item)
    assert queue.shape[queue.queue_dim] == 6
    assert queue[-1] == item


def test_torch_queue_push_multiple_dimensions():
    queue = TorchQueue(torch.randn(3, 5), 10, queue_dim=1)
    item = torch.randn(3, 1)
    queue = queue.push(item)
    assert queue.shape[queue.queue_dim] == 6
    torch.testing.assert_close(queue[:, -1], item.squeeze(1))


def test_torch_queue_pop():
    item = torch.randn(10)
    queue = TorchQueue(item, 10)
    popped, queue = queue.pop()
    assert queue.shape[queue.queue_dim] == 9
    assert queue[-1] == item[-1]
    assert popped == item[0]


def test_torch_queue_indexing():
    item = torch.randn(10)
    queue = TorchQueue(item, 10)
    assert queue.shape[queue.queue_dim] == 10
    assert queue[-1] == item[-1]
    assert queue[0] == item[0]
    assert isinstance(queue[0], torch.Tensor)
    assert isinstance(queue[-1], torch.Tensor)
    assert isinstance(queue[5], torch.Tensor)
    assert not isinstance(queue[0], TorchQueue)
    assert not isinstance(queue[-1], TorchQueue)
    assert not isinstance(queue[5], TorchQueue)


def test_torch_queue_multiple_push():
    queue = TorchQueue(torch.randn(5), 10)
    for i in range(15):
        item = torch.randn(1)
        queue = queue.push(item)
        assert queue.shape[queue._queue_dim] == min(5 + i + 1, 10)
        assert queue[-1] == item


def test_torch_queue_multiple_push_multiple_dimensions():
    queue = TorchQueue(torch.randn(3, 5), 10, queue_dim=1)
    for i in range(15):
        item = torch.randn(3, 1)
        queue = queue.push(item)
        assert queue.shape[queue._queue_dim] == min(5 + i + 1, 10)
        torch.testing.assert_close(queue[:, -1], item.squeeze(1))


def test_torch_queue_multiple_pop():
    queue = TorchQueue(torch.randn(10), 10)
    last_value = queue[-1]
    for i in range(5):
        oldest_value = queue[0]
        item, queue = queue.pop()
        assert queue.shape[queue._queue_dim] == 10 - i - 1
        assert last_value == queue[-1]
        assert oldest_value == item


def test_torch_queue_full():
    queue = TorchQueue(torch.randn(10), 10)
    item = torch.randn(1)
    queue = queue.push(item)
    assert queue.shape[queue._queue_dim] == 10
    assert queue[-1] == item


def test_torch_queue_empty():
    queue = TorchQueue(torch.randn(0), 10)
    with pytest.raises(IndexError):
        queue.pop()
    assert queue.shape[queue._queue_dim] == 0


def test_torch_queue_push_to_capacity_and_overflow():
    capacity = 5
    queue = TorchQueue(torch.randn(capacity - 2), capacity)
    # Push until capacity
    while not queue.is_full:
        item = torch.randn(1)
        queue = queue.push(item)
    assert queue.is_full
    assert queue.shape[queue.queue_dim] == capacity

    # Record current first and push one more to cause overflow
    first_before_value = queue[0].item()
    new_item = torch.randn(1)
    queue = queue.push(new_item)
    assert queue.is_full
    assert queue.shape[queue.queue_dim] == capacity
    # Oldest dropped, newest at the end
    assert queue[-1] == new_item
    assert queue[0].item() != first_before_value


def test_torch_queue_clone_preserves_attrs_and_data():
    data = torch.arange(6.0)
    queue = TorchQueue(data, queue_size=6, queue_dim=0)
    clone_q = queue.clone()
    assert isinstance(clone_q, TorchQueue)
    assert clone_q.queue_size == queue.queue_size
    assert clone_q.queue_dim == queue.queue_dim
    torch.testing.assert_close(clone_q, queue)


def test_torch_queue_to_device_roundtrip():
    queue = TorchQueue(torch.randn(4), 10)
    # CPU no-op
    q_cpu = queue.to("cpu")
    assert isinstance(q_cpu, TorchQueue)
    assert q_cpu.queue_size == queue.queue_size
    assert q_cpu.queue_dim == queue.queue_dim
    # Optional CUDA
    if torch.cuda.is_available():
        q_cuda = queue.to("cuda")
        assert isinstance(q_cuda, TorchQueue)
        assert q_cuda.device.type == "cuda"
        q_back = q_cuda.to("cpu")
        assert isinstance(q_back, TorchQueue)
        assert q_back.device.type == "cpu"


def test_torch_queue_queue_dim_behavior():
    # Queue along dim=1 with 2D data
    data = torch.randn(3, 4)  # (batch=3, time=4)
    capacity = 5
    q = TorchQueue(data, queue_size=capacity, queue_dim=1)
    assert not q.is_full
    # Push a time step for all batch items: shape (3,1)
    item = torch.randn(3, 1)
    q = q.push(item)
    assert q.shape[1] == 5
    assert q.is_full
    # Overflow by one more push
    item2 = torch.randn(3, 1)
    q = q.push(item2)
    assert q.shape[1] == 5
    assert q.is_full
    # Pop oldest time step and check shapes
    oldest, q2 = q.pop()
    assert oldest.shape == torch.Size([3])
    assert q2.shape[1] == 4
