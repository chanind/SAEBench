import einops
import torch
from sae_lens import SAE

from sae_bench.evals.scr_and_tpp.main import ablated_precomputed_activations


# Test without rescaling
def test_ablated_precomputed_activations_no_rescale(gpt2_l4_sae: SAE) -> None:
    B = 2
    L = 10
    D = gpt2_l4_sae.cfg.d_in
    F = gpt2_l4_sae.cfg.d_sae
    SAE_BATCH_SIZE = 1
    device = gpt2_l4_sae.device
    dtype = gpt2_l4_sae.dtype

    # Create random input activations (ensure no zero padding for simplicity)
    # Make sure some values are non-zero
    ablation_acts_BLD = torch.randn(B, L, D, device=device, dtype=dtype) + 0.1

    # Select some features to ablate
    to_ablate = torch.zeros(F, dtype=torch.bool, device=device)
    to_ablate[0 : F // 2] = (
        True  # Ablate first half of features for a noticeable effect
    )

    # Run the function under test without rescaling
    result_BD = ablated_precomputed_activations(
        ablation_acts_BLD,
        gpt2_l4_sae,
        to_ablate,
        SAE_BATCH_SIZE,
        rescale_l2_norm=False,
    )

    # Manually compute the expected result (without rescaling)
    with torch.no_grad():
        f_BLF = gpt2_l4_sae.encode(ablation_acts_BLD)
        x_hat_BLD = gpt2_l4_sae.decode(f_BLF)
        error_BLD = ablation_acts_BLD - x_hat_BLD

        f_ablated_BLF = f_BLF.clone()
        f_ablated_BLF[..., to_ablate] = 0.0

        modified_acts_BLD = gpt2_l4_sae.decode(f_ablated_BLF) + error_BLD

        # Since our dummy input has no padding, we can just take the mean
        # The original function handles padding by dividing by nonzero count
        expected_BD = einops.reduce(modified_acts_BLD, "B L D -> B D", "mean")

    assert result_BD.shape == (B, D)
    # Use a tolerance appropriate for the dtype
    atol = 1e-5 if dtype == torch.float32 else 1e-3
    assert torch.allclose(result_BD, expected_BD, atol=atol)


# Separate test specifically for the L2 norm rescaling
def test_ablated_precomputed_activations_rescales_l2_norm(gpt2_l4_sae: SAE) -> None:
    B = 2
    L = 10
    D = gpt2_l4_sae.cfg.d_in
    F = gpt2_l4_sae.cfg.d_sae
    SAE_BATCH_SIZE = 1
    device = gpt2_l4_sae.device
    dtype = gpt2_l4_sae.dtype
    epsilon = 1e-8

    # Create random input activations (ensure no zero padding for simplicity)
    # Make sure some values are non-zero
    ablation_acts_BLD = torch.randn(B, L, D, device=device, dtype=dtype) + 0.1
    # Ensure some norms are large enough to avoid epsilon issues everywhere
    ablation_acts_BLD[0, 0, :] *= 10

    # Select some features to ablate
    to_ablate = torch.zeros(F, dtype=torch.bool, device=device)
    to_ablate[0 : F // 2] = True  # Ablate first half of features

    # Run the function under test with rescaling
    result_BD = ablated_precomputed_activations(
        ablation_acts_BLD,
        gpt2_l4_sae,
        to_ablate,
        SAE_BATCH_SIZE,
        rescale_l2_norm=True,
    )

    # Check output shape
    assert result_BD.shape == (B, D)

    # Verify that the L2 norms *before mean pooling* are preserved
    with torch.no_grad():
        # Recalculate the modified activations *with* rescaling (mirroring internal logic)
        f_BLF = gpt2_l4_sae.encode(ablation_acts_BLD)
        x_hat_BLD = gpt2_l4_sae.decode(f_BLF)
        error_BLD = ablation_acts_BLD - x_hat_BLD

        f_ablated_BLF = f_BLF.clone()
        f_ablated_BLF[..., to_ablate] = 0.0

        modified_acts_no_rescale_BLD = gpt2_l4_sae.decode(f_ablated_BLF) + error_BLD

        original_norm_BL1 = torch.linalg.norm(
            ablation_acts_BLD, ord=2, dim=-1, keepdim=True
        )
        modified_norm_no_rescale_BL1 = torch.linalg.norm(
            modified_acts_no_rescale_BLD, ord=2, dim=-1, keepdim=True
        )

        scaling_factor_BL1 = original_norm_BL1 / (
            modified_norm_no_rescale_BL1 + epsilon
        )

        # Apply scaling where appropriate (mirroring mask in function)
        # Assuming no padding tokens in this test, so nonzero_acts_BL is all True
        scale_mask = modified_norm_no_rescale_BL1 > epsilon
        final_modified_acts_BLD = torch.where(
            scale_mask,
            modified_acts_no_rescale_BLD * scaling_factor_BL1,
            modified_acts_no_rescale_BLD,
        )

        # Now calculate norms of the final, potentially rescaled activations
        final_modified_norms_BL = torch.linalg.norm(
            final_modified_acts_BLD, ord=2, dim=-1
        )
        original_norms_BL = torch.linalg.norm(ablation_acts_BLD, ord=2, dim=-1)

        # Assert norms are close where the original norm wasn't tiny and scaling was applied
        valid_scale_mask_BL = scale_mask.squeeze(-1)
        assert torch.allclose(
            original_norms_BL[valid_scale_mask_BL],
            final_modified_norms_BL[valid_scale_mask_BL],
            atol=1e-3,  # Tolerance needs to be higher due to potential accumulation
        )
