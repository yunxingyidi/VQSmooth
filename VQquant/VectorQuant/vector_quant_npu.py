import torch

try:
    import torch_npu  # noqa: F401
except ModuleNotFoundError:
    torch_npu = None


@torch.no_grad()
def dequant_forward(
    w_indices: torch.Tensor,
    w_codebook: torch.Tensor,
    w_dq: torch.Tensor,
    n: int,
) -> None:
    del n  # Kept for API compatibility with the legacy extension.

    if w_indices.dim() != 2:
        raise ValueError(f"Expected w_indices to be 2D, got {tuple(w_indices.shape)}")
    if w_codebook.dim() != 2:
        raise ValueError(f"Expected w_codebook to be 2D, got {tuple(w_codebook.shape)}")
    if w_dq.dim() != 2:
        raise ValueError(f"Expected w_dq to be 2D, got {tuple(w_dq.shape)}")

    n_groups, out_features = w_indices.shape
    in_features, codebook_size = w_codebook.shape

    if in_features % n_groups != 0:
        raise ValueError(
            f"in_features {in_features} must be divisible by n_groups {n_groups}"
        )
    if w_dq.shape != (in_features, out_features):
        raise ValueError(
            f"Expected w_dq shape {(in_features, out_features)}, got {tuple(w_dq.shape)}"
        )

    group_width = in_features // n_groups

    codebook = w_codebook.reshape(n_groups, group_width, codebook_size)
    gather_index = (
        w_indices.to(dtype=torch.long)
        .unsqueeze(1)
        .expand(n_groups, group_width, out_features)
    )
    dequantized = torch.gather(codebook, dim=2, index=gather_index)
    w_dq.copy_(dequantized.reshape(in_features, out_features))
