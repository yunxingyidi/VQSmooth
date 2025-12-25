import sys, os
sys.path.append(os.path.dirname(__file__))
import time

import torch
import torch.nn as nn

import transformers

from smooth_vq import *
from modelutils import *
from vector_quant import *
import sys, os
sys.path.append(os.path.dirname(__file__))

def get_llama(model):
    import torch

    def skip(*_, **__):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    model.seqlen = 2048
    return model

@torch.no_grad()
def llama_sequential(model, dataloader, dev, args):
    print("Quantization....")
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass

    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    print("Creating quantizer...")
    QClass = lambda: VectorQuantizer(
        sub_vector=args.sub_vector,
        assignment_chunk_size=args.assignment_chunk_size,
        kmeans_iters=args.kmeans_iters,
        codebook_width=args.codebook_width,
    )
    quantizers = {}

    for i in range(len(layers)):
        layer = layers[i].to(dev)

        full = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
                ["self_attn.o_proj"],
                ["mlp.up_proj", "mlp.gate_proj"],
                ["mlp.down_proj"],
            ]
        else:
            sequential = [[k for k in list(full.keys()) if "block_sparse_moe.gate" not in k]]

        for names in sequential:
            subset = {n: full[n] for n in names}
            smoothvq = {}
            for name in subset:
                smoothvq[name] = SmoothVQ(subset[name])
                smoothvq[name].quantizer = QClass()
                smoothvq[name].quantizer.configure(codebook_width=args.codebook_width)

            def add_batch(name):
                def tmp(_, inp, out):
                    smoothvq[name].add_batch(inp[0].detach().clone(),
            out.detach().clone())

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(
                    inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids
                )[0]
            for h in handles:
                h.remove()
            for name in subset:
                print(i, name)
                smoothvq[name].fasterquant(
                    use_vq=args.use_vq,
                    name=name
                )
                quantizers["model.layers.%d.%s" % (i, name)] = smoothvq[name].quantizer
                smoothvq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids
            )[0]

        layers[i] = layer.cpu()
        del layer
        del smoothvq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache

    return quantizers

@torch.no_grad()
def llama_eval(model, testenc, dev):

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    for i in range(len(layers)):
        print(i)
        # layer = layers[i].to(dev)
        layer = layers[i].to(dev)
        for j in range(nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids
            )[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    with open("ppl_log.txt", "a") as f:
        f.write(f"{ppl.item()}\n")

    print(ppl.item())
    model.config.use_cache = use_cache