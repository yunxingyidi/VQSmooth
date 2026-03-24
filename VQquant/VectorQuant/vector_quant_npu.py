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
) -> None:

    if w_indices.dim() != 2:
        raise ValueError(f"Expected w_indices to be 2D, got {tuple(w_indices.shape)}")
    if w_codebook.dim() != 2:
        raise ValueError(f"Expected w_codebook to be 2D, got {tuple(w_codebook.shape)}")
    if w_dq.dim() != 2:
        raise ValueError(f"Expected w_dq to be 2D, got {tuple(w_dq.shape)}")

    n_groups, out_features = w_indices.shape
    codebook_size, in_features = w_codebook.shape

    if in_features % n_groups != 0:
        raise ValueError(
            f"in_features {in_features} must be divisible by n_groups {n_groups}"
        )
    if w_dq.shape != (out_features, in_features):
        raise ValueError(
            f"Expected w_dq shape {(out_features, in_features)}, got {tuple(w_dq.shape)}"
        )

    group_width = in_features // n_groups

    codebook = w_codebook.reshape(codebook_size, n_groups, group_width).permute(1, 2, 0)
    gather_index = (
        w_indices.to(dtype=torch.long)
        .unsqueeze(1)
        .expand(n_groups, group_width, out_features)
    )
    dequantized = torch.gather(codebook, dim=2, index=gather_index)
    w_dq.copy_(dequantized.permute(2, 0, 1).reshape(out_features, in_features))
