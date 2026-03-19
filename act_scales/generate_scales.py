import torch
try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import argparse
import sys, os
sys.path.append(".")
from calibration import get_act_scales


def resolve_device(device):
    if device == "auto":
        if torch_npu is not None and hasattr(torch, "npu") and torch.npu.is_available():
            return torch.device("npu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if device == "npu":
        if torch_npu is None or not hasattr(torch, "npu") or not torch.npu.is_available():
            raise RuntimeError(
                "Requested NPU, but torch_npu is not installed or no NPU is available."
            )
        return torch.device("npu")

    return torch.device(device)


def build_model_and_tokenizer(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    kwargs = {"torch_dtype": torch.float16}
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model = model.to(device)
    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="facebook/opt-1.3b", help="model name"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="act_scales/llama-2-7b.pt",
        help="where to save the act scales",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="dataset/val.jsonl.zst",
        help="location of the calibration dataset, we use the validation set of the Pile dataset",
    )
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument(
        "--device",
        type=str,
        default="npu",
        choices=["auto", "npu", "cuda", "cpu"],
        help="device to run calibration on",
    )
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    model, tokenizer = build_model_and_tokenizer(args.model, device)

    if not os.path.exists(args.dataset_path):
        print(f"Cannot find the dataset at {args.dataset_path}")
        print("Please download the Pile dataset and put the validation set at the path")
        print(
            "You can download the validation dataset of the Pile at https://huggingface.co/datasets/mit-han-lab/pile-val-backup/resolve/main/val.jsonl.zst"
        )
        raise FileNotFoundError

    act_scales = get_act_scales(
        model, tokenizer, args.dataset_path, args.num_samples, args.seq_len
    )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(act_scales, args.output_path)


if __name__ == "__main__":
    main()
