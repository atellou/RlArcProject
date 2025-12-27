"""Test FSDP compatibility with RlArcWorldViTARC module.

This test verifies that FSDP wrapping works correctly with the ViTARC encoder,
particularly ensuring that embedding layers are not corrupted during sharding.
"""

import logging

import pytest
import torch
import torch.distributed as dist
from tensordict import TensorDict

from rlarcworld.common.configs import RlArcWorldViTARCConfig
from rlarcworld.common.enums import StateKeys
from rlarcworld.models.nn_modules import RlArcWorldViTARC

logger = logging.getLogger(__name__)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for FSDP")
def test_fsdp_vitarc_single_gpu():
    """Test FSDP wrapping of RlArcWorldViTARC on single GPU.

    This test verifies that:
    1. FSDP can wrap the model without errors
    2. Embedding layers maintain correct 2-D shape
    3. Forward pass works correctly
    4. Gradients flow properly
    """
    # Skip if distributed not available
    if not dist.is_available():
        pytest.skip("torch.distributed not available")

    # Initialize process group for single-GPU FSDP
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="tcp://localhost:29500",
            world_size=1,
            rank=0,
        )

    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy

        batch_size = 2
        grid_size = 8
        color_values = 11
        num_examples = 2

        # Create config with all features enabled (matching production)
        config = RlArcWorldViTARCConfig(
            grid_size=grid_size,
            d_model=64,
            d_ff=128,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
            use_ope=True,
            use_temporal=True,
            use_examples=True,
            use_index_grid=True,
            color_values=color_values,
        )

        # Create model
        model = RlArcWorldViTARC(config).cuda()

        # Configure FSDP with same policy as production
        fsdp_wrap_policy = ModuleWrapPolicy(
            {
                torch.nn.TransformerEncoderLayer,
                torch.nn.TransformerDecoderLayer,
            }
        )

        # Wrap with FSDP
        model = FSDP(
            model,
            device_id=torch.cuda.current_device(),
            use_orig_params=True,
            auto_wrap_policy=fsdp_wrap_policy,
        )

        logger.info("FSDP wrapping successful")

        # Verify embedding layer shapes are preserved
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Embedding):
                weight = module.weight
                assert weight.ndim == 2, (
                    f"Embedding {name} weight is not 2-D: shape={weight.shape}"
                )
                logger.info(f"Embedding {name} has correct 2-D shape: {weight.shape}")

        # Create input state with all required fields
        state = TensorDict(
            {
                StateKeys.GRID.value: torch.randint(
                    0,
                    color_values,
                    (batch_size, 1, grid_size, grid_size),
                    device="cuda",
                ),
                StateKeys.INITIAL.value: torch.randint(
                    0,
                    color_values,
                    (batch_size, 1, grid_size, grid_size),
                    device="cuda",
                ),
                StateKeys.LAST_GRID.value: torch.randint(
                    0,
                    color_values,
                    (batch_size, 1, grid_size, grid_size),
                    device="cuda",
                ),
                StateKeys.EXAMPLES.value: torch.randint(
                    0,
                    color_values,
                    (batch_size, num_examples, 2, grid_size, grid_size),
                    device="cuda",
                ),
                StateKeys.INDEX.value: torch.randint(
                    0, grid_size, (batch_size, 1, grid_size, grid_size), device="cuda"
                ),
            }
        )

        # Forward pass
        logger.info("Running forward pass...")
        output = model(state)

        assert output.hidden_states is not None
        assert output.hidden_states.shape[0] == batch_size
        assert output.hidden_states.shape[2] == config.d_model
        logger.info(
            f"Forward pass successful, output shape: {output.hidden_states.shape}"
        )

        # Backward pass to verify gradients flow
        logger.info("Running backward pass...")
        loss = output.hidden_states.mean()
        loss.backward()

        # Check gradients exist
        grad_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_count += 1

        assert grad_count > 0, "No gradients found after backward pass"
        logger.info(f"Backward pass successful, {grad_count} parameters have gradients")

        # Verify embedding gradients specifically
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Embedding):
                assert module.weight.grad is not None, (
                    f"Embedding {name} has no gradient"
                )
                assert module.weight.grad.ndim == 2, (
                    f"Embedding {name} gradient is not 2-D"
                )
                logger.info(
                    f"Embedding {name} gradient shape: {module.weight.grad.shape}"
                )

        logger.info("FSDP test passed successfully")

    finally:
        # Cleanup
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for FSDP")
def test_fsdp_vitarc_embedding_shape_preservation():
    """Test that embedding layers maintain 2-D shape after FSDP wrapping.

    This is a regression test for the 'weight must be 2-D' error.
    """
    if not dist.is_available():
        pytest.skip("torch.distributed not available")

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="tcp://localhost:29501",
            world_size=1,
            rank=0,
        )

    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy

        grid_size = 8
        color_values = 11

        config = RlArcWorldViTARCConfig(
            grid_size=grid_size,
            d_model=64,
            d_ff=128,
            num_layers=2,
            num_heads=4,
            color_values=color_values,
        )

        model = RlArcWorldViTARC(config).cuda()

        # Record original embedding shapes
        original_shapes = {}
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Embedding):
                original_shapes[name] = module.weight.shape
                logger.info(f"Original {name} shape: {module.weight.shape}")

        # Wrap with FSDP
        fsdp_wrap_policy = ModuleWrapPolicy(
            {
                torch.nn.TransformerEncoderLayer,
                torch.nn.TransformerDecoderLayer,
            }
        )

        model = FSDP(
            model,
            device_id=torch.cuda.current_device(),
            use_orig_params=True,
            auto_wrap_policy=fsdp_wrap_policy,
        )

        # Verify shapes are preserved
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Embedding):
                current_shape = module.weight.shape
                assert name in original_shapes, (
                    f"Embedding {name} not found in original shapes"
                )
                assert current_shape == original_shapes[name], (
                    f"Embedding {name} shape changed: ",
                    f"{original_shapes[name]} -> {current_shape}",
                )
                assert len(current_shape) == 2, (
                    f"Embedding {name} is not 2-D: {current_shape}"
                )
                logger.info(f"Verified {name} shape preserved: {current_shape}")

        logger.info("All embedding shapes preserved correctly")

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    # Run tests directly for debugging
    logging.basicConfig(level=logging.INFO)
    test_fsdp_vitarc_single_gpu()
    test_fsdp_vitarc_embedding_shape_preservation()
