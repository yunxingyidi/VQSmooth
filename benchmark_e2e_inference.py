import argparse
import gc
import json
import pickle
import sys
import time

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

from transformers import AutoModelForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig

from Smoothquant.group_quant import quantize_model
from Smoothquant.kv_cache import QuantizedDynamicCache

sys.path.append(".")
from VQquant.datautils import load_tokenizer
from VQquant.llama import get_llama


def parse_args():
    parser = argparse.ArgumentParser("Benchmark quantized LLM end-to-end inference")
    parser.add_argument("--model", type=str, required=True, help="Path to the HF model")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the quantized checkpoint")
    parser.add_argument("--device", type=str, default="auto", help="Execution device, e.g. npu:0")
    parser.add_argument(
        "--benchmark-mode",
        type=str,
        default="compare",
        choices=["compare", "bf16", "quantized"],
        help="Benchmark BF16 baseline, quantized model, or both",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain what vector quantization is in simple terms.",
        help="Prompt used for generation",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Optional text file to read prompt from",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Maximum new tokens to generate")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Number of warmup runs")
    parser.add_argument("--benchmark-runs", type=int, default=3, help="Number of timed runs")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Enable KV cache during decoding. Default is off for compatibility with the quantized path.",
    )
    parser.add_argument(
        "--print-generated-text",
        action="store_true",
        help="Print generated text for the last timed run of each mode",
    )
    parser.add_argument("--fake-quant", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--smooth", action="store_true")
    parser.add_argument("--group-quantize", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--eval-only", action="store_true", default=True)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seed-data", type=int, default=0)
    parser.add_argument("--codebook-width", type=int, default=8)
    parser.add_argument("--residual_group", type=int, default=4)
    parser.add_argument("--sub-vector", type=int, default=2)
    parser.add_argument("--use-vq", action="store_true")
    parser.add_argument("--true-sequential", action="store_true")
    parser.add_argument("--kmeans-iters", type=int, default=10)
    parser.add_argument("--assignment-chunk-size", type=int, default=None)
    parser.add_argument("--vq-devices", type=str, default=None)
    args = parser.parse_args()
    args.eval_only = True
    return args


def resolve_device(device):
    if device == "auto":
        if torch_npu is not None and hasattr(torch, "npu") and torch.npu.is_available():
            return torch.device("npu:0")
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    if device.startswith("npu"):
        if torch_npu is None or not hasattr(torch, "npu") or not torch.npu.is_available():
            raise RuntimeError("Requested NPU, but torch_npu is not installed or no NPU is available.")
        return torch.device(device)

    return torch.device(device)


def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize(device)


def empty_device_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.empty_cache()


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
            f"missing_keys={incompat.missing_keys}, unexpected_keys={incompat.unexpected_keys}"
        )


def read_prompt(args):
    if args.prompt_file is None:
        return args.prompt
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        return f.read()


def load_base_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model = get_llama(model)
    model.eval()
    return model


def build_bf16_model(args, device):
    t0 = time.perf_counter()
    model = load_base_model(args.model)
    if next(model.parameters()).device != device:
        model = model.to(device)
    sync_device(device)
    return model, {"bf16_model_load_time_s": time.perf_counter() - t0}


def build_quantized_model(args, device):
    t0 = time.perf_counter()
    model = load_base_model(args.model)
    base_model_load_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    model = quantize_model(model, args)
    load_groupql_checkpoint(model, args.ckpt)
    if next(model.parameters()).device != device:
        model = model.to(device)
    sync_device(device)
    restore_time = time.perf_counter() - t0
    return model, {
        "base_model_load_time_s": base_model_load_time,
        "quantized_restore_time_s": restore_time,
    }


@torch.no_grad()
def generate_once(model, tokenizer, input_ids, device, max_new_tokens, use_cache):
    generated = input_ids.clone()
    eos_token_id = tokenizer.eos_token_id
    past_key_values = QuantizedDynamicCache() if use_cache else None
    prefill_time = 0.0
    decode_time = 0.0

    for step in range(max_new_tokens):
        has_cache = use_cache and past_key_values is not None and past_key_values.get_seq_length(0) > 0
        step_input = generated[:, -1:] if has_cache else generated
        sync_device(device)
        start = time.perf_counter()
        if use_cache and past_key_values is not None:
            outputs = model(step_input, use_cache=True, past_key_values=past_key_values)
            past_key_values = outputs.past_key_values
        else:
            outputs = model(step_input, use_cache=use_cache)
            if use_cache:
                past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        sync_device(device)
        elapsed = time.perf_counter() - start

        if step == 0:
            prefill_time = elapsed
        else:
            decode_time += elapsed

        generated = torch.cat([generated, next_token], dim=-1)
        if eos_token_id is not None and torch.all(next_token == eos_token_id):
            break

    generated_tokens = generated.shape[1] - input_ids.shape[1]
    total_time = prefill_time + decode_time
    return {
        "prefill_time_s": prefill_time,
        "decode_time_s": decode_time,
        "total_time_s": total_time,
        "generated_tokens": int(generated_tokens),
        "first_token_latency_s": prefill_time,
        "tokens_per_s": generated_tokens / total_time if total_time > 0 else 0.0,
        "decode_tokens_per_s": max(generated_tokens - 1, 0) / decode_time if decode_time > 0 else 0.0,
        "output_ids": generated,
    }


def summarize_runs(metrics):
    numeric_keys = [
        "prefill_time_s",
        "decode_time_s",
        "total_time_s",
        "generated_tokens",
        "first_token_latency_s",
        "tokens_per_s",
        "decode_tokens_per_s",
    ]
    summary = {}
    for key in numeric_keys:
        values = [item[key] for item in metrics]
        summary[key] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
    return summary


def benchmark_model(label, model, tokenizer, input_ids, device, args):
    print(f"[{label}] warmup start")
    for idx in range(args.warmup_runs):
        _ = generate_once(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            device=device,
            max_new_tokens=args.max_new_tokens,
            use_cache=args.use_cache,
        )
        print(f"[{label}] warmup {idx + 1}/{args.warmup_runs} complete")

    metrics = []
    last_output_ids = None
    print(f"[{label}] timed runs start")
    for idx in range(args.benchmark_runs):
        result = generate_once(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            device=device,
            max_new_tokens=args.max_new_tokens,
            use_cache=args.use_cache,
        )
        last_output_ids = result.pop("output_ids")
        metrics.append(result)
        print(
            f"[{label}] run {idx + 1}/{args.benchmark_runs}: "
            f"first_token={result['first_token_latency_s']:.4f}s, "
            f"total={result['total_time_s']:.4f}s, "
            f"gen_tokens={result['generated_tokens']}, "
            f"tokens/s={result['tokens_per_s']:.4f}"
        )

    return summarize_runs(metrics), last_output_ids


def release_model(model, device):
    del model
    gc.collect()
    empty_device_cache(device)


def build_comparison(bf16_summary, quant_summary):
    return {
        "first_token_latency_ratio_quant_over_bf16": (
            quant_summary["first_token_latency_s"]["mean"] / bf16_summary["first_token_latency_s"]["mean"]
            if bf16_summary["first_token_latency_s"]["mean"] > 0
            else None
        ),
        "total_time_ratio_quant_over_bf16": (
            quant_summary["total_time_s"]["mean"] / bf16_summary["total_time_s"]["mean"]
            if bf16_summary["total_time_s"]["mean"] > 0
            else None
        ),
        "tokens_per_s_ratio_quant_over_bf16": (
            quant_summary["tokens_per_s"]["mean"] / bf16_summary["tokens_per_s"]["mean"]
            if bf16_summary["tokens_per_s"]["mean"] > 0
            else None
        ),
        "decode_tokens_per_s_ratio_quant_over_bf16": (
            quant_summary["decode_tokens_per_s"]["mean"] / bf16_summary["decode_tokens_per_s"]["mean"]
            if bf16_summary["decode_tokens_per_s"]["mean"] > 0
            else None
        ),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.set_device(device)

    prompt = read_prompt(args)

    t0 = time.perf_counter()
    tokenizer = load_tokenizer(args.model)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer_load_time = time.perf_counter() - t0

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc.input_ids.to(device)

    print(f"Using device: {device}")
    print(f"Benchmark mode: {args.benchmark_mode}")
    print(f"Prompt tokens: {input_ids.shape[1]}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Use cache: {args.use_cache}")
    print(f"Tokenizer load time: {tokenizer_load_time:.4f}s")

    report = {
        "benchmark_mode": args.benchmark_mode,
        "device": str(device),
        "prompt_tokens": int(input_ids.shape[1]),
        "max_new_tokens": args.max_new_tokens,
        "use_cache": args.use_cache,
        "tokenizer_load_time_s": tokenizer_load_time,
    }

    if args.benchmark_mode in ("compare", "bf16"):
        bf16_model, bf16_load = build_bf16_model(args, device)
        print(f"BF16 model load time: {bf16_load['bf16_model_load_time_s']:.4f}s")
        bf16_summary, bf16_output_ids = benchmark_model("bf16", bf16_model, tokenizer, input_ids, device, args)
        report["bf16"] = {**bf16_load, "summary": bf16_summary}
        if args.print_generated_text and bf16_output_ids is not None:
            print("[bf16] generated text:")
            print(tokenizer.decode(bf16_output_ids[0], skip_special_tokens=True))
        release_model(bf16_model, device)

    if args.benchmark_mode in ("compare", "quantized"):
        quant_model, quant_load = build_quantized_model(args, device)
        print(f"Quantized base model load time: {quant_load['base_model_load_time_s']:.4f}s")
        print(f"Quantized model restore time: {quant_load['quantized_restore_time_s']:.4f}s")
        quant_summary, quant_output_ids = benchmark_model("quantized", quant_model, tokenizer, input_ids, device, args)
        report["quantized"] = {**quant_load, "summary": quant_summary}
        if args.print_generated_text and quant_output_ids is not None:
            print("[quantized] generated text:")
            print(tokenizer.decode(quant_output_ids[0], skip_special_tokens=True))
        release_model(quant_model, device)

    if args.benchmark_mode == "compare":
        report["comparison"] = build_comparison(
            report["bf16"]["summary"],
            report["quantized"]["summary"],
        )

    print("Summary:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
