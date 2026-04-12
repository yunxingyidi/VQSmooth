from urllib.parse import uses_query
import pickle

import torch
try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from Smoothquant.smooth import smooth_lm
from Smoothquant.group_quant import (
    quantize_model,
    GroupQLinear,
    prewarm_weight_quant_runtime_cache,
)
import argparse
import time
import sys
sys.path.append(".")
from VQquant.llama import get_llama, llama_eval, llama_sequential
from VQquant.datautils import get_loaders, get_test_loader
from VQquant.modelutils import *

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument(
    "--act_scales_path",
    type=str,
    default="act_scales/llama-2-7b.pt",
)
parser.add_argument("--nsamples", type=int, default=128)
parser.add_argument("--smooth", action="store_true")
parser.add_argument("--group-quantize", action="store_true")
parser.add_argument("--save-model", action="store_true")
parser.add_argument("--eval-only", action="store_true")
parser.add_argument("--fake-quant", action="store_true")
parser.add_argument("--weight-quant", action="store_true")
parser.add_argument("--profile-eval", action="store_true")
parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
parser.add_argument(
        "--codebook-width", type=int, default=8, help="Bitwidth for codebook quantization"
    )
parser.add_argument(
        "--residual_group", type=int, default=4, help="Bitwidth for codebook quantization"
    )
parser.add_argument("--sub-vector", type=int, default=2, help="Dimensionality of VQ")
parser.add_argument(
        "--use-vq", action="store_true", help="If set, use VQ (multi-dim non-uniform) quantization"
    )
parser.add_argument(
        "--true-sequential", action="store_true", help="Whether to run in true sequential model."
    )
parser.add_argument("--kmeans-iters", type=int, default=10)
parser.add_argument(
        "--assignment-chunk-size",
        type=int,
        default=None,
        help="Chunk assignment step for better memory management",
    )
parser.add_argument(
        "--dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4"],
        help="Where to extract calibration data from.",
    )
parser.add_argument(
    "--ckpt",
    type=str,
    default="vq_smooth_cb8_sv2_int8.pt",
    help="Path to GroupQLinear checkpoint",
)
parser.add_argument(
    "--device",
    type=str,
    default="auto",
    help="device to run evaluation on",
)
parser.add_argument(
    "--vq-devices",
    type=str,
    default=None,
    help="comma-separated devices for VQ quantization, e.g. npu:0,npu:1; evaluation still runs on --device",
)

args = parser.parse_args()


def resolve_device(device):
    if device == "auto":
        if torch_npu is not None and hasattr(torch, "npu") and torch.npu.is_available():
            return torch.device("npu:0")
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    if device.startswith("npu"):
        if torch_npu is None or not hasattr(torch, "npu") or not torch.npu.is_available():
            raise RuntimeError(
                "Requested NPU, but torch_npu is not installed or no NPU is available."
            )
        return torch.device(device)

    return torch.device(device)


device = resolve_device(args.device)
DEV = device
print(f"Using device: {device}")


def load_groupql_checkpoint(model, ckpt_path):
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except pickle.UnpicklingError:
        with torch.serialization.safe_globals([LlamaConfig]):
            ckpt = torch.load(ckpt_path, map_location="cpu")

    state_dict = ckpt["model"] if "model" in ckpt else ckpt

    incompat = model.load_state_dict(state_dict, strict=True)

    if incompat.missing_keys or incompat.unexpected_keys:
        raise RuntimeError(
            "Checkpoint structure does not match the current quantized model. "
            f"missing_keys={incompat.missing_keys}, "
            f"unexpected_keys={incompat.unexpected_keys}"
        )

    print("[GroupQLinear] checkpoint loaded")

def build_ckpt_name(args):
    if args.use_vq:
        parts = ["vq"]
    else:
        parts = []

    parts.append("smooth" if args.smooth else "nosmooth")
    parts.append(f"cb{args.codebook_width}")
    parts.append(f"sv{args.sub_vector}")
    parts.append("wq8" if args.weight_quant else "wbf16")

    return "_".join(parts) + ".pt"


@torch.no_grad()
def prewarm_weight_quant_caches(model, device):
    warmed = 0
    for module in model.modules():
        if isinstance(module, GroupQLinear) and getattr(module, "weight_quant", False):
            prewarm_weight_quant_runtime_cache(module, device)
            warmed += 1
    return warmed

model = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.bfloat16
)
if not args.eval_only:
    model = model.to(device)
model = get_llama(model)
model.eval()

if args.eval_only:
    testloader = get_test_loader(
        args.dataset, model=args.model, seqlen=model.seqlen
    )
else:
    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

if args.eval_only:
    model = quantize_model(
        model,
        args
    )
    load_groupql_checkpoint(model, args.ckpt)
else:
    if args.smooth:
        print("Smooth quantize...")
        act_scales = torch.load(args.act_scales_path)
        smooth_lm(model, act_scales, args.alpha)
    if args.group_quantize:
        # if args.codebook_width < 16:
        #     quantizers = llama_sequential(model, dataloader, DEV, args)
        print("Vector quantize...")
        model = quantize_model(
            model,
            args
        )
    if args.save_model:
        print("Saving model...")
        ckpt_name = build_ckpt_name(args)
        checkpoint = {
            "model": model.state_dict(),
            "config": model.config,  # 如果你是 HF LLaMA
        }
        torch.save(checkpoint, ckpt_name)

if device.type == "npu" and hasattr(torch, "npu"):
    torch.npu.set_device(device)
if next(model.parameters()).device != device:
    model = model.to(device)

if args.weight_quant:
    prewarm_start = time.perf_counter()
    warmed_layers = prewarm_weight_quant_caches(model, device)
    prewarm_seconds = time.perf_counter() - prewarm_start
    print(
        f"Weight-quant prewarm done: layers={warmed_layers}, "
        f"elapsed={prewarm_seconds:.3f}s"
    )

print(f"Running llama_eval on device: {device}")
eval_start = time.perf_counter()
llama_eval(model, testloader, DEV, profile=args.profile_eval)
eval_seconds = time.perf_counter() - eval_start

testenc = testloader.input_ids
total_tokens = testenc.numel()
tok_per_s = total_tokens / eval_seconds if eval_seconds > 0 else float("inf")
print(f"Eval elapsed: {eval_seconds:.3f}s")
print(f"Eval throughput: {tok_per_s:.2f} tok/s (total_tokens={total_tokens})")
