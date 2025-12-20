import pytest
import torch

from rlarcworld.arc_dataset import ArcDataset, ArcSampleTransformer

# ---------- Fixtures ----------


@pytest.fixture()
def grid_size():
    return (15, 10)


@pytest.fixture()
def n_examples():
    return 5


@pytest.fixture()
def padding_constant_value():
    return -1


@pytest.fixture()
def transform(grid_size, padding_constant_value):
    return ArcSampleTransformer(
        grid_size,
        padding_constant_value=padding_constant_value,
        zero_based_correction=0,
    )


@pytest.fixture()
def arc_dataset_dir():
    return "tests/test_data"


@pytest.fixture()
def dataset_one(arc_dataset_dir):
    return ArcDataset(arc_dataset_dir=arc_dataset_dir, keep_in_memory=False)


@pytest.fixture()
def dataset_two(arc_dataset_dir):
    return ArcDataset(arc_dataset_dir=arc_dataset_dir, keep_in_memory=True)


# ---------- Tests: ArcSampleTransformer ----------


def test_padding_method(transform, grid_size, padding_constant_value):
    task = {"input": torch.ones((5, 5))}
    padded = transform.pad_to_size(task["input"])
    assert isinstance(padded, torch.Tensor)
    assert padded.shape == grid_size
    assert torch.min(padded) == padding_constant_value
    assert torch.max(padded) == 1
    assert torch.max(padded[:5, :5]) == 1
    assert torch.min(padded[:5, :5]) == 1
    with pytest.raises(AssertionError):
        transform.pad_to_size(torch.ones((30, 30)))


def test_concat_unsqueeze(transform):
    task = {"grid_one": torch.ones((5, 5)), "grid_two": torch.zeros((5, 5))}
    unsqzed = transform.concat_unsqueeze(**task)
    assert isinstance(unsqzed, torch.Tensor)
    assert unsqzed.shape == (
        2,
        *task["grid_one"].shape,
    )
    task = {"grid_one": torch.ones((5, 5)), "grid_two": torch.zeros((6, 6))}
    with pytest.raises(AssertionError):
        transform.concat_unsqueeze(**task)


def test_extraction_transformed_transform(transform, n_examples, grid_size):
    dataset_t = ArcDataset(
        arc_dataset_dir="tests/test_data",
        keep_in_memory=False,
        transform=transform,
    )
    assert len(dataset_t) == 4
    for d in dataset_t:
        torch.testing.assert_close(d["examples"].shape[1:], [2] + [*grid_size])
        torch.testing.assert_close(d["task"]["input"].shape, [*grid_size])
        torch.testing.assert_close(d["task"]["output"].shape, [*grid_size])
        assert isinstance(d["examples"], torch.Tensor)
        assert isinstance(d["task"]["input"], torch.Tensor)
        assert isinstance(d["task"]["output"], torch.Tensor)


# ---------- Tests: ArcDataset ----------


def test_loader(arc_dataset_dir):
    grid_size = (10, 10)
    n_examples = 5
    batch_size = 2
    dataset_t = ArcDataset(
        arc_dataset_dir=arc_dataset_dir,
        keep_in_memory=False,
        transform=ArcSampleTransformer(grid_size, n_examples, zero_based_correction=0),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset_t, batch_size=batch_size, shuffle=True, num_workers=0
    )
    for sample_batched in dataloader:
        torch.testing.assert_close(sample_batched["examples"].shape[0], batch_size)
        torch.testing.assert_close(
            sample_batched["examples"].shape[2:], [2] + [*grid_size]
        )
        torch.testing.assert_close(
            sample_batched["task"]["input"].shape, [batch_size] + [*grid_size]
        )
        torch.testing.assert_close(
            sample_batched["task"]["input"].shape,
            sample_batched["task"]["output"].shape,
        )


def test_extraction(dataset_one, dataset_two):
    assert len(dataset_one) == 4
    assert len(dataset_two) == 4
    for d_1, d_2 in zip(dataset_one, dataset_two):
        assert all(d_1) == all(d_2)


def test_extraction_transformed(arc_dataset_dir):
    grid_size = (10, 10)
    dataset_t = ArcDataset(
        arc_dataset_dir=arc_dataset_dir,
        keep_in_memory=False,
        transform=ArcSampleTransformer(grid_size, zero_based_correction=0),
    )
    assert len(dataset_t) == 4
    for d in dataset_t:
        torch.testing.assert_close(d["examples"].shape[1:], [2] + [*grid_size])
        torch.testing.assert_close(d["task"]["input"].shape, [*grid_size])
        torch.testing.assert_close(d["task"]["output"].shape, [*grid_size])
        assert isinstance(d["examples"], torch.Tensor)
        assert isinstance(d["task"]["input"], torch.Tensor)
        assert isinstance(d["task"]["output"], torch.Tensor)


def test_getitem(dataset_one, arc_dataset_dir):
    # Valid index
    item = dataset_one[0]
    assert isinstance(item, dict)
    assert "examples" in item
    assert "task" in item
    assert len(item["examples"]) == 2
    assert len(item["task"]) == 2

    # Invalid index
    with pytest.raises(IndexError):
        dataset_one[4]
    with pytest.raises(IndexError):
        dataset_one.open_file(f"{arc_dataset_dir}/test_2.json", test_index=999)


def test_open_file(dataset_one, arc_dataset_dir):
    result = dataset_one.open_file(f"{arc_dataset_dir}/test1.json")

    assert isinstance(result, dict)
    assert len(result) == 2
    assert "train" in result
    assert "test" in result

    assert len(result["train"]) == 2
    for item in result["train"]:
        assert "input" in item
        assert "output" in item
        assert isinstance(item["input"], torch.Tensor)
        assert isinstance(item["output"], torch.Tensor)

    assert isinstance(result["test"], list)
    assert len(result["test"]) == 2
    for item in result["test"]:
        assert "input" in item
        assert "output" in item
        assert isinstance(item["input"], torch.Tensor)
        assert isinstance(item["output"], torch.Tensor)

    torch.testing.assert_close(
        result["train"][0]["input"], torch.tensor([[1, 2], [3, 4]])
    )
    torch.testing.assert_close(
        result["train"][0]["output"], torch.tensor([[5, 6], [7, 8]])
    )

    torch.testing.assert_close(
        result["test"][0]["input"], torch.tensor([[1, 2], [3, 4]])
    )
    torch.testing.assert_close(
        result["test"][0]["output"], torch.tensor([[5, 6], [7, 8]])
    )


def test_open_file_with_test_index(arc_dataset_dir, dataset_one):
    result = dataset_one.open_file(f"{arc_dataset_dir}/test_2.json", test_index=1)

    assert "train" in result
    assert "test" in result

    assert len(result["train"]) == 2
    torch.testing.assert_close(
        result["train"][0]["input"], torch.tensor([[1, 2], [3, 4]])
    )
    torch.testing.assert_close(
        result["train"][0]["output"], torch.tensor([[5, 6], [7, 8]])
    )
    torch.testing.assert_close(
        result["train"][1]["input"], torch.tensor([[9, 10], [0, 1]])
    )
    torch.testing.assert_close(
        result["train"][1]["output"], torch.tensor([[2, 3], [4, 5]])
    )

    assert isinstance(result["test"], dict)
    torch.testing.assert_close(result["test"]["input"], torch.tensor([[8, 9], [10, 0]]))
    torch.testing.assert_close(result["test"]["output"], torch.tensor([[1, 2], [3, 4]]))


def test_extraction_multiple_dirs():
    """Test that the transformed dataset works with multiple directories."""
    dataset_t = ArcDataset(
        arc_dataset_dir=["tests/test_data", "tests/test_data2"],
        keep_in_memory=False,
    )
    assert len(dataset_t) == 8


# ---------- Negative tests for ArcSampleTransformer ----------


def test_transform_init_invalid_output_size_type():
    # output_size must be a tuple per implementation
    with pytest.raises(AssertionError):
        ArcSampleTransformer(output_size=10)
    with pytest.raises(AssertionError):
        ArcSampleTransformer(output_size=[10, 10])


def test_pad_to_size_invalid_rank(transform):
    # 1D tensor should fail (expects 2D)
    with pytest.raises(AssertionError):
        transform.pad_to_size(torch.ones(10))
    # 3D tensor should fail
    with pytest.raises(AssertionError):
        transform.pad_to_size(torch.ones(1, 5, 5))


def test_transform_call_with_invalid_example_grid_rank(transform):
    # Build sample where one example has 1D tensors instead of 2D
    sample = {
        "examples": [
            {"input": torch.ones(5), "output": torch.ones(5)},
        ],
        "task": {"input": torch.ones(2, 2), "output": torch.ones(2, 2)},
    }
    with pytest.raises(AssertionError):
        transform(sample)


def test_transform_call_with_missing_example_keys(transform):
    # Missing 'input'/'output' keys should error during concat
    sample = {
        "examples": [
            {"inp": torch.ones(2, 2), "out": torch.ones(2, 2)},
        ],
        "task": {"input": torch.ones(2, 2), "output": torch.ones(2, 2)},
    }
    with pytest.raises(KeyError):
        transform(sample)


def test_zero_based_correction_applied_to_examples():
    """Test that zero_based_correction is applied to examples, not just task grids.

    This ensures padding values (-1) become 0 and all color values are positive.
    """
    grid_size = (5, 5)
    padding_value = -1
    correction = 1

    transform = ArcSampleTransformer(
        output_size=grid_size,
        padding_constant_value=padding_value,
        zero_based_correction=correction,
    )

    # Create sample with small grids that will be padded
    sample = {
        "examples": [
            {"input": torch.zeros(2, 2), "output": torch.ones(2, 2)},
        ],
        "task": {"input": torch.zeros(3, 3), "output": torch.ones(3, 3)},
    }

    result = transform(sample)

    # Task grids should have correction applied: min should be 0 + 1 = 1 (from padding)
    # Original zeros become 1, padding (-1) becomes 0
    assert result["task"]["input"].min().item() == padding_value + correction  # 0
    assert result["task"]["output"].min().item() == padding_value + correction  # 0

    # Examples should ALSO have correction applied (this was the bug)
    # Without the fix, examples would have min = -1 (padding not corrected)
    assert result["examples"].min().item() == padding_value + correction  # 0
    assert result["examples"].min().item() >= 0, (
        f"Examples contain negative values ({result['examples'].min().item()}), "
        "zero_based_correction not applied to examples"
    )
