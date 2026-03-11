from urllib.parse import uses_query

from transformers import AutoTokenizer, AutoModelForCausalLM
from Smoothquant.smooth import smooth_lm
from Smoothquant.group_quant import quantize_model
import argparse
import sys
sys.path.append(".")
from VQquant.llama import get_llama, llama_eval, llama_sequential
from VQquant.datautils import get_loaders
from VQquant.modelutils import *
import torch

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
    default="vq_ckpt.pt",
    help="Path to GroupQLinear checkpoint",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda:0",
    help="Single device used for llama_eval.",
)
parser.add_argument(
    "--quant-device-map",
    type=str,
    default="auto",
    help="Device map used during smooth/VQ stage. Use 'none' for single-device loading.",
)

args = parser.parse_args()
eval_device = torch.device(args.device)

def load_groupql_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    state_dict = ckpt["model"] if "model" in ckpt else ckpt

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=True,
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

    return "_".join(parts) + ".pt"

def build_model_for_quant_stage(args):
    load_kwargs = {"torch_dtype": torch.bfloat16}
    if args.quant_device_map and args.quant_device_map.lower() != "none":
        load_kwargs["device_map"] = args.quant_device_map
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model = get_llama(model)
    model.eval()
    return model


def move_to_single_eval_device(model, args):
    is_quantized = args.group_quantize or args.eval_only
    print(f"Rebuild model on {args.device} for llama_eval...")
    cpu_state = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    del model
    torch.cuda.empty_cache()

    eval_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
    )
    eval_model = get_llama(eval_model)
    if is_quantized:
        eval_args = argparse.Namespace(**vars(args))
        eval_args.eval_only = True
        eval_model = quantize_model(eval_model, eval_args)

    eval_model.load_state_dict(cpu_state, strict=True)
    eval_model = eval_model.to(args.device)
    eval_model.eval()
    return eval_model


model = build_model_for_quant_stage(args)
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

model = move_to_single_eval_device(model, args)
llama_eval(model, testloader, eval_device)
