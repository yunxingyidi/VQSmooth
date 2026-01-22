import numpy as np
import torch
from torch import nn
import time

def get_assignments(X, centroids, chunk_size=None):
    """
    X: N x K x D
    centroids: N x K x D
    """
    if chunk_size is None:
        X_chunks = [X]
        centroids_chunks = [centroids]
    else:
        X_chunks = torch.split(X, chunk_size, dim=0)  # N_chunk x R x D
        centroids_chunks = torch.split(centroids, chunk_size, dim=0)  # N_chunk x K x D

    # centroids = centroids.unsqueeze(1) # N x 1 x K x D

    assignments_chunks = []
    for X_chunk, cent_chunk in zip(X_chunks, centroids_chunks):
        # X_chunk: [N_chunk, R, D], cent_chunk: [N_chunk, K, D]
        dist = ((X_chunk.unsqueeze(2) - cent_chunk.unsqueeze(1)) ** 2).sum(-1)  # [N_chunk, R, K]
        assignments_chunks.append(dist.argmin(-1))  # [N_chunk, R]

    assignments = torch.cat(assignments_chunks, dim=0)
    return assignments

def vq_quantize(X, quantizer, centroids=None, fake_quant=False):
    assert len(X.shape) == 2
    R, C = X.shape

    sub_vector = quantizer.sub_vector
    X = X.reshape(R, -1, sub_vector)  # R x C//D x D
    X = X.permute(1, 0, 2)
    if centroids is None:
        centroids = quantizer.all_centroids[-1]  # N x K x D

    idx = get_assignments(
        X, centroids, chunk_size=quantizer.assignment_chunk_size
    )  # N x R
    # below, idx expanded to N x K x D
    if fake_quant:
        values = torch.gather(centroids, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, sub_vector))
        values = values.permute(1, 0, 2)
        return values.reshape(R, C), idx
        # return shapes: N x R x D, G x N
    else:
        return idx

def kmeans_m_step_3(
    centroids: torch.Tensor,
    n_centroids: int,
    assignments: torch.LongTensor,
    X: torch.Tensor,
):
    """
    Fully batch KMeans M-step using scatter_add to avoid OOM.

    Args:
        X:        [N, R, D] tensor of samples
        centroids:[N, K, D] tensor to be updated
        assignments:[N, R] long tensor of cluster indices
    """
    N, R, D = X.shape
    K = n_centroids
    device = X.device
    dtype = X.dtype

    # Flatten batch dimension to use scatter_add in one shot
    # offset each batch by K to avoid collisions
    batch_offset = torch.arange(N, device=device) * K  # [N]
    assignments_flat = assignments + batch_offset.unsqueeze(1)  # [N, R]
    X_flat = X.reshape(N * R, D)  # [N*R, D]
    assignments_flat = assignments_flat.reshape(-1)  # [N*R]

    # Sum points per cluster (all batches together)
    clusters_sum = torch.zeros((N * K, D), dtype=dtype, device=device)
    clusters_sum = clusters_sum.index_add(0, assignments_flat, X_flat)

    # Count points per cluster
    counts = torch.zeros(N * K, dtype=dtype, device=device)
    ones = torch.ones(N * R, dtype=dtype, device=device)
    counts = counts.index_add(0, assignments_flat, ones)

    # Reshape back to [N, K, D] and normalize
    clusters_sum = clusters_sum.view(N, K, D)
    counts = counts.view(N, K)
    norm = 1.0 / torch.clamp(counts, min=1.0)
    centroids.copy_(clusters_sum * norm.unsqueeze(-1))

def kmeans_vq(
    X,
    centroids,
    iters=10,
    assignment_chunk_size=None,
):
    n_centroids = centroids.shape[1]
    for iter in range(iters):
        # E-step
        assignments = get_assignments(
            X, centroids, chunk_size=assignment_chunk_size
        )

        # M-step: gather all values for each centroid and compute means
        # Centroids is shape N x K x D; assignments is shape G x N
        kmeans_m_step_3(centroids, n_centroids, assignments, X)

def kpp_parallel_sampled(data: torch.Tensor, k: int):
    N, R, D = data.shape
    all_init = []

    # 根据 N 分 batch，避免一次性生成过大矩阵
    if N < 16:
        split_data = data.split(1)
    elif N < 64:
        split_data = data.split(4)
    else:
        split_data = data.split(8)

    for batch in split_data:
        B = batch.shape[0]  # 当前 batch 的矩阵数量
        init = torch.zeros((B, k, D), dtype=torch.float16, device=batch.device)
        init[:, 0, :] = batch[:, 0, :]  # 每个矩阵第一个点作为初始中心

        # 每个 batch 的距离矩阵
        D2 = torch.zeros(B, k, R, dtype=torch.float16, device=batch.device)
        all_dists = torch.cdist(batch.to(torch.float16), batch.to(torch.float16), p=2)  # [B, R, R]
        D2[:, 0] = all_dists[:, 0]

        for i in range(1, k):
            # 选最近的距离
            dists = D2[:, :i].amin(dim=1)  # [B, R]
            probs = (dists / dists.sum(dim=1, keepdims=True)).cumsum(dim=1)  # [B, R]

            v = torch.rand(B, 1, device=batch.device)
            idx = torch.clip(torch.searchsorted(probs, v), 0, R - 1).unsqueeze(-1)  # [B, 1]

            # 更新 D2 和 init
            D2[:, i:i+1, :] = torch.gather(all_dists, 1, idx.expand(B, 1, R))
            init[:, i:i+1, :] = torch.gather(batch, 1, idx.expand(B, 1, D))

        all_init.append(init)

    return torch.cat(all_init, dim=0)  # [N, k, D]

class VectorQuantizer(nn.Module):
    def __init__(
        self,
        sub_vector=2,
        n_subsample=100000,
        assignment_chunk_size=None,
        kmeans_iters=10,
        codebook_width=None
    ):
        super().__init__()
        self.sub_vector = sub_vector
        self.n_centroids = None
        self.all_centroids = []
        self.kpp_subsamples = n_subsample
        self.assignment_chunk_size = assignment_chunk_size
        self.kmeans_iters = kmeans_iters
        self.codebook_width = codebook_width

    def configure(self, codebook_width, **_):
        self.codebook_width = int(codebook_width)
        self.n_centroids = int(2**self.codebook_width)

    def get_centroids(self, X: torch.Tensor, weight=True):
        assert weight
        assert len(X.shape) == 2
        R, C = X.shape
        assert C % self.sub_vector == 0

        X = X.reshape(R, -1, self.sub_vector)  # R x C//D x D
        X = X.permute(1, 0, 2)
        centroids = kpp_parallel_sampled(X, self.n_centroids)

        extra_args = {}

        kmeans_vq(
            X,
            centroids,
            iters=self.kmeans_iters,
            assignment_chunk_size=self.assignment_chunk_size,
            **extra_args,
        )

        self.all_centroids.append(centroids)