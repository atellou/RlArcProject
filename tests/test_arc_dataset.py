from rlarcworld.arc_dataset import ArcDataset, ArcSampleTransformer
import json
import torch
import unittest


# Create a test ArcSampleTransformer instance
class TestArcSampleTransformer(unittest.TestCase):
    def setUp(self):
        self.grid_size = (15, 10)
        self.n_examples = 5
        self.padding_constant_value = -1
        self.transform = ArcSampleTransformer(
            self.grid_size,
            self.n_examples,
            padding_constant_value=self.padding_constant_value,
            zero_based_correction=0,
        )

    def test_padding_method(self):
        """
        Test the sample method of ArcSampleTransformer.
        """
        # Test with a valid task
        task = {"input": torch.ones((5, 5))}
        padded = self.transform.pad_to_size(task["input"])
        assert isinstance(padded, torch.Tensor)
        assert padded.shape == self.grid_size
        assert torch.min(padded) == self.padding_constant_value
        assert torch.max(padded) == 1
        assert torch.max(padded[:5, :5]) == 1
        assert torch.min(padded[:5, :5]) == 1
        with self.assertRaises(AssertionError):
            self.transform.pad_to_size(torch.ones((30, 30)))

    def test_concat_unsqueezed(self):
        """
        Test the concat_unsqueezed method of ArcSampleTransformer.
        """
        # Test with a valid task
        task = {"grid_one": torch.ones((5, 5)), "grid_two": torch.zeros((5, 5))}
        unsqzed = self.transform.concat_unsqueezed(**task)
        assert isinstance(unsqzed, torch.Tensor)
        assert unsqzed.shape == (
            2,
            *task["grid_one"].shape,
        )
        task = {"grid_one": torch.ones((5, 5)), "grid_two": torch.zeros((6, 6))}
        with self.assertRaises(AssertionError):
            self.transform.concat_unsqueezed(**task)

    def test_extraction_transformed(self):
        dataset_t = ArcDataset(
            arc_dataset_dir="tests/test_data",
            keep_in_memory=False,
            transform=self.transform,
        )
        assert len(dataset_t) == 4
        for d in dataset_t:
            torch.testing.assert_close(
                d["examples"].shape, [self.n_examples, 2] + [*self.grid_size]
            )
            torch.testing.assert_close(d["task"]["input"].shape, [*self.grid_size])
            torch.testing.assert_close(d["task"]["output"].shape, [*self.grid_size])
            assert isinstance(d["examples"], torch.Tensor)
            assert isinstance(d["task"]["input"], torch.Tensor)
            assert isinstance(d["task"]["output"], torch.Tensor)


# Create a test ArcDataset instance
class TestArcDataset(unittest.TestCase):
    def setUp(self):
        self.arc_dataset_dir = "tests/test_data"
        self.dataset_one = ArcDataset(
            arc_dataset_dir=self.arc_dataset_dir, keep_in_memory=False
        )
        self.dataset_two = ArcDataset(
            arc_dataset_dir=self.arc_dataset_dir, keep_in_memory=True
        )

    def test_loader(self):
        grid_size = (10, 10)
        n_examples = 5
        batch_size = 2
        dataset_t = ArcDataset(
            arc_dataset_dir=self.arc_dataset_dir,
            keep_in_memory=False,
            transform=ArcSampleTransformer(
                grid_size, n_examples, zero_based_correction=0
            ),
        )
        dataloader = torch.utils.data.DataLoader(
            dataset_t, batch_size=batch_size, shuffle=True, num_workers=0
        )
        for sample_batched in dataloader:
            torch.testing.assert_close(
                sample_batched["examples"].shape,
                [batch_size, n_examples, 2] + [*grid_size],
            )
            torch.testing.assert_close(
                sample_batched["task"]["input"].shape,
                [batch_size] + [*grid_size],
            )
            torch.testing.assert_close(
                sample_batched["task"]["input"].shape,
                sample_batched["task"]["output"].shape,
            )

    def test_extraction(self):
        assert len(self.dataset_one) == 4
        assert len(self.dataset_two) == 4
        for d_1, d_2 in zip(self.dataset_one, self.dataset_two):
            assert all(d_1) == all(d_2)

    def test_extraction_transformed(self):
        grid_size = (10, 10)
        n_examples = 5
        dataset_t = ArcDataset(
            arc_dataset_dir=self.arc_dataset_dir,
            keep_in_memory=False,
            transform=ArcSampleTransformer(
                grid_size, n_examples, zero_based_correction=0
            ),
        )
        assert len(dataset_t) == 4
        for d in dataset_t:
            torch.testing.assert_close(
                d["examples"].shape, [n_examples, 2] + [*grid_size]
            )
            torch.testing.assert_close(d["task"]["input"].shape, [*grid_size])
            torch.testing.assert_close(d["task"]["output"].shape, [*grid_size])
            assert isinstance(d["examples"], torch.Tensor)
            assert isinstance(d["task"]["input"], torch.Tensor)
            assert isinstance(d["task"]["output"], torch.Tensor)

    def test_getitem(self):
        """
        Test the __getitem__ method of ArcDataset.
        """
        # Test with a valid index
        item = self.dataset_one[0]
        assert isinstance(item, dict)
        assert "examples" in item
        assert "task" in item
        assert len(item["examples"]) == 2
        assert len(item["task"]) == 2

        # Test with an invalid index
        with self.assertRaises(IndexError):
            self.dataset_one[4]
            self.dataset_one.open_file(
                f"{self.arc_dataset_dir}/test_2.json", test_index=999
            )

    def test_open_file(self):
        """
        Test open_file method when test_index is None.
        """
        # Mock the open function to return our mock file content
        result = self.dataset_one.open_file(f"{self.arc_dataset_dir}/test1.json")

        # Assert the structure of the returned dictionary
        assert isinstance(result, dict)
        assert len(result) == 2
        assert "train" in result
        assert "test" in result

        # Check the train data
        assert len(result["train"]) == 2
        for item in result["train"]:
            assert "input" in item
            assert "output" in item
            assert isinstance(item["input"], torch.Tensor)
            assert isinstance(item["output"], torch.Tensor)

        # Check the test data
        assert isinstance(result["test"], list)
        assert len(result["test"]) == 2
        for item in result["test"]:
            assert "input" in item
            assert "output" in item
            assert isinstance(item["input"], torch.Tensor)
            assert isinstance(item["output"], torch.Tensor)

        # Verify the content of the first train item
        torch.testing.assert_close(
            result["train"][0]["input"], torch.tensor([[1, 2], [3, 4]])
        )
        torch.testing.assert_close(
            result["train"][0]["output"], torch.tensor([[5, 6], [7, 8]])
        )

        # Verify the content of the test item
        torch.testing.assert_close(
            result["test"][0]["input"], torch.tensor([[17, 18], [19, 20]])
        )
        torch.testing.assert_close(
            result["test"][0]["output"], torch.tensor([[21, 22], [23, 24]])
        )

    def test_open_file_with_test_index(self):
        """
        Test open_file method when test_index is provided.
        """
        result = self.dataset_one.open_file(
            f"{self.arc_dataset_dir}/test_2.json", test_index=1
        )

        # Assertions
        assert "train" in result
        assert "test" in result

        # Check train data
        assert len(result["train"]) == 2
        torch.testing.assert_close(
            result["train"][0]["input"], torch.tensor([[1, 2], [3, 4]])
        )
        torch.testing.assert_close(
            result["train"][0]["output"], torch.tensor([[5, 6], [7, 8]])
        )
        torch.testing.assert_close(
            result["train"][1]["input"], torch.tensor([[9, 10], [11, 12]])
        )
        torch.testing.assert_close(
            result["train"][1]["output"], torch.tensor([[13, 14], [15, 16]])
        )

        # Check test data
        assert isinstance(result["test"], dict)
        torch.testing.assert_close(
            result["test"]["input"], torch.tensor([[25, 26], [27, 28]])
        )
        torch.testing.assert_close(
            result["test"]["output"], torch.tensor([[29, 30], [31, 32]])
        )


if __name__ == "__main__":
    unittest.main()
