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

args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.bfloat16, device_map="auto"
)
model = get_llama(model)
model.eval()
dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

if args.eval_only:
    state_dict = torch.load(
        "/home/zhangtr/WorkSpace/models/quantized_model_state_dict.pt",
        map_location="cpu"
    )
    model.load_state_dict(state_dict)
    model = model.cuda()
    model = quantize_model(
        model,
        act_quant="per_token",
        quantize_bmm_input=True,
    )
else:
    if args.smooth:
        act_scales = torch.load(args.act_scales_path)
        smooth_lm(model, act_scales, args.alpha)
    if args.group_quantize:
        model = quantize_model(
            model,
            act_quant="per_token",
            quantize_bmm_input=True,
        )
    # if args.codebook_width < 16:
    #     quantizers = llama_sequential(model, dataloader, DEV, args)

    if args.save_model:
        torch.save(
            model.state_dict(),
            "quant_llama_state.pt"
        )

llama_eval(model, testloader, DEV)


