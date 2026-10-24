#!/usr/bin/env python3
"""boxfit: which local-inference box your own workload stalls on, before you buy it.

    python3 boxfit.py config.json my-run.log [--only spark,m5max] [--explain] [--json]

Reads a Hugging Face config.json and a log from the machine you run today (Ollama
--verbose output or server.log, llama.cpp llama-server or llama-cli, mlx_lm.generate
--verbose), then prints per box: does the model fit, seconds to the first token,
tokens per second, and a year of your use owned against rented.

Standard library only. No GPU, no network, no API key. Tested 2026-09-28.
"""
import argparse
import json
import os
import re
import statistics
import sys
from datetime import datetime

VERSION = "0.1"
GB = 1e9
GIB = 2 ** 30

# ----------------------------------------------------------------------------- model

def _get(cfg, *names, default=None):
    """First present key, looking in text_config (multimodal wrappers) and then the top."""
    scopes = [cfg.get("text_config") or {}, cfg]
    for n in names:
        for s in scopes:
            if n in s and s[n] is not None:
                return s[n]
    return default


class Model:
    """Parameter and byte accounting for one decoder, from config.json alone."""

    def __init__(self, cfg, name="model"):
        self.cfg = cfg
        self.name = name
        H = _get(cfg, "hidden_size", "d_model", "n_embd")
        L = _get(cfg, "num_hidden_layers", "n_layer", "num_layers")
        if not H or not L:
            raise ValueError("config.json has no hidden_size or num_hidden_layers")
        nh = _get(cfg, "num_attention_heads", "n_head")
        nkv = _get(cfg, "num_key_value_heads", default=nh)
        hd = _get(cfg, "head_dim", default=(H // nh if nh else None))
        V = _get(cfg, "vocab_size", default=32000)
        tied = bool(_get(cfg, "tie_word_embeddings", default=False))
        n_exp = _get(cfg, "num_local_experts", "num_experts", "n_routed_experts", default=0) or 0
        k_exp = _get(cfg, "num_experts_per_tok", "experts_per_token", "num_experts_per_token", default=0) or 0
        i_dense = _get(cfg, "intermediate_size", default=4 * H)
        i_moe = _get(cfg, "moe_intermediate_size", default=i_dense)
        i_shared = _get(cfg, "shared_expert_intermediate_size", default=0) or 0
        n_shared = _get(cfg, "n_shared_experts", "num_shared_experts", default=1 if i_shared else 0) or 0
        if n_shared and not i_shared:
            i_shared = i_moe
        window = _get(cfg, "sliding_window", default=None)
        types = _get(cfg, "layer_types", default=None) or ["full_attention"] * L
        gate_q = bool(_get(cfg, "attn_output_gate", default=False))

        # which layers carry experts
        dense_layers = set()
        k_dense = _get(cfg, "first_k_dense_replace", default=0) or 0
        dense_layers |= set(range(k_dense))
        dense_layers |= set(_get(cfg, "mlp_only_layers", default=[]) or [])
        step = _get(cfg, "decoder_sparse_step", default=1) or 1
        if step > 1:
            dense_layers |= {i for i in range(L) if (i + 1) % step != 0}
        moe_layers = [i for i in range(L) if n_exp and i not in dense_layers]

        # linear-attention (Gated DeltaNet style) dimensions
        lk_h = _get(cfg, "linear_num_key_heads", default=0) or 0
        lk_d = _get(cfg, "linear_key_head_dim", default=0) or 0
        lv_h = _get(cfg, "linear_num_value_heads", default=0) or 0
        lv_d = _get(cfg, "linear_value_head_dim", default=0) or 0
        conv_k = _get(cfg, "linear_conv_kernel_dim", default=4) or 4

        attn = 0.0            # attention and linear-attention projection params (all layers)
        self.layers = []      # (kind, window) for the KV and attention-FLOP accounting
        self.linear_state_bytes = 0.0
        self.linear_flops = 0.0
        for i in range(L):
            kind = types[i] if i < len(types) else "full_attention"
            if "linear" in kind and lk_h and lv_h:
                kdim, vdim = lk_h * lk_d, lv_h * lv_d
                attn += H * (2 * kdim + vdim) + H * vdim + vdim * H + 2 * H * lv_h + (2 * kdim + vdim) * conv_k
                self.layers.append(("linear", None))
                self.linear_state_bytes += lv_h * lk_d * lv_d * 4          # fp32 recurrent state
                self.linear_flops += 4 * lv_h * lk_d * lv_d
            else:
                q_out = nh * hd * (2 if gate_q else 1)
                attn += H * q_out + 2 * H * nkv * hd + nh * hd * H
                self.layers.append(("sliding", window) if ("sliding" in kind and window) else ("full", None))
        mlp_dense = 3 * H * i_dense * (L - len(moe_layers))
        experts = 3 * H * i_moe * n_exp * len(moe_layers)
        shared = 3 * H * i_shared * n_shared * len(moe_layers)
        router = H * n_exp * len(moe_layers)
        embed = V * H
        head = 0 if tied else V * H

        self.H, self.L, self.nh, self.nkv, self.hd, self.V = H, L, nh, nkv, hd, V
        self.tied = tied
        self.n_exp, self.k_exp = n_exp, k_exp
        self.expert_params = float(experts)
        self.active_expert_params = experts * (k_exp / n_exp) if n_exp else 0.0
        self.attn_params = float(attn)
        self.dense_mlp_params = float(mlp_dense)
        self.shared_params = float(shared)
        self.router_params = float(router)
        self.embed_params = float(embed)
        self.head_params = float(head if head else embed)   # tied heads still get read
        self.total_params = experts + attn + mlp_dense + shared + router + embed + head
        # read on every token: everything except the unused experts and the embedding table
        self.active_params = (self.active_expert_params + attn + mlp_dense + shared + router
                              + self.head_params)
        self.full_layers = sum(1 for k, _ in self.layers if k == "full")
        self.sliding_layers = [w for k, w in self.layers if k == "sliding"]

    # -- bytes
    def kv_bytes_per_token(self, kv_bytes=2):
        """KV cache added per token of context on every full-attention layer."""
        return self.full_layers * 2 * self.nkv * self.hd * kv_bytes

    def kv_read_bytes(self, depth, kv_bytes=2):
        per = 2 * self.nkv * self.hd * kv_bytes
        b = self.full_layers * per * depth
        b += sum(per * min(depth, w) for w in self.sliding_layers)
        return b + self.linear_state_bytes

    # -- flops
    def mm_flops_per_token(self):
        return 2 * self.active_params + self.linear_flops

    def attn_flops_per_depth(self):
        """Attention FLOPs added per token for each token already in context (full layers)."""
        return self.full_layers * 4 * self.nh * self.hd

    def sliding_attn_flops(self, depth):
        return sum(4 * self.nh * self.hd * min(depth, w) for w in self.sliding_layers)


# ----------------------------------------------------------------------------- quantization

# bits per weight including block scales, by llama.cpp tensor type or file type
TYPE_BITS = {
    "f32": 32.0, "f16": 16.0, "bf16": 16.0, "q8_0": 8.5, "q6_k": 6.5625, "q5_k": 5.5, "q5_0": 5.5,
    "q5_1": 6.0, "q4_k": 4.5, "q4_0": 4.5, "q4_1": 5.0, "q3_k": 3.4375, "q2_k": 2.625,
    "iq4_xs": 4.25, "iq4_nl": 4.5, "iq3_xxs": 3.0625, "iq2_xxs": 2.0625, "mxfp4": 4.25,
}
FILE_TYPE_BITS = [  # (pattern in llama.cpp "file type" line, expert bits, dense bits)
    (r"mxfp4", 4.25, 8.5), (r"q4_k.*m", 4.85, 4.85), (r"q4_k.*s", 4.58, 4.58),
    (r"q5_k.*m", 5.69, 5.69), (r"q6_k", 6.56, 6.56), (r"q8_0", 8.5, 8.5),
    (r"q3_k.*m", 3.91, 3.91), (r"iq4_xs", 4.25, 4.25), (r"q4_0", 4.5, 4.5),
    (r"bf16|f16", 16.0, 16.0),
]
PRESETS = {  # --quant names: bits per weight for (experts, dense layers, output head)
    "mxfp4-moe": (4.25, 8.5, 8.5), "nvfp4": (4.5, 8.0, 4.5), "q4_k_m": (4.85, 4.85, 4.85),
    "q8_0": (8.5, 8.5, 8.5), "f16": (16.0, 16.0, 16.0), "mlx-4bit": (4.5, 4.5, 4.5), "mlx-8bit": (8.5, 8.5, 8.5),
}


def quant_from_config(cfg):
    """({expert, dense, head} bits, why) from a config's own quantization block, or None."""
    q = cfg.get("quantization_config") or cfg.get("quantization") or {}
    if not q:
        return None
    method = str(q.get("quant_method", q.get("mode", ""))).lower()
    if method == "mxfp4":
        return dict(expert=4.25, dense=16.0, head=16.0), "config.json: mxfp4 experts, the rest unconverted"
    if method == "modelopt" and q.get("config_groups"):
        bits = dict(expert=16.0, dense=16.0, head=16.0)
        for g in q["config_groups"].values():
            w = g.get("weights") or {}
            b = float(w.get("num_bits", 16))
            if b == 4 and w.get("group_size"):
                b += 8.0 / w["group_size"]              # one FP8 scale per group
            tg = g.get("targets", [])
            if any(".experts" in x or "shared_expert" in x for x in tg):
                bits["expert"] = b
            if any("attn" in x for x in tg):
                bits["dense"] = b
            if any(x.endswith("lm_head") for x in tg):
                bits["head"] = b
        return bits, "config.json: modelopt groups (experts %.2f, attention %.2f, head %.2f bits)" % (
            bits["expert"], bits["dense"], bits["head"])
    if "bits" in q:  # mlx-community configs: {"group_size": 64, "bits": 4}
        b = float(q["bits"]) + 32.0 / float(q.get("group_size", 64))
        return dict(expert=b, dense=b, head=b), "config.json: MLX %s-bit, group %s" % (q["bits"], q.get("group_size", 64))
    return None


# ----------------------------------------------------------------------------- log

class Workload:
    def __init__(self):
        self.requests = []          # dicts: prompt, gen, prompt_tps, gen_tps
        self.file_bytes = None
        self.bpw = None
        self.log_params = None
        self.tensor_types = {}
        self.file_type = None
        self.times = []

    def add(self, **r):
        self.requests.append(r)


_num = r"([0-9][0-9,]*\.?[0-9]*)"


def _f(s):
    return float(str(s).replace(",", ""))


def _dur(s):
    """Ollama durations: 1.234s, 450.2ms, 1m2.5s."""
    s = s.strip()
    total = 0.0
    for val, unit in re.findall(r"([0-9.]+)\s*(ms|µs|us|s|m|h)", s):
        v = float(val)
        total += {"ms": v / 1e3, "µs": v / 1e6, "us": v / 1e6, "s": v, "m": v * 60, "h": v * 3600}[unit]
    return total


def parse_log(text):
    w = Workload()
    # llama.cpp model loader lines (also printed inside Ollama's server.log)
    m = re.search(r"file size\s*=\s*" + _num + r"\s*(GiB|GB|MiB)(?:\s*\(\s*" + _num + r"\s*BPW\))?", text)
    if m:
        v = _f(m.group(1))
        w.file_bytes = v * {"GiB": GIB, "GB": GB, "MiB": 2 ** 20}[m.group(2)]
        w.bpw = _f(m.group(3)) if m.group(3) else None
    m = re.search(r"model params\s*=\s*" + _num + r"\s*([BM])", text)
    if m:
        w.log_params = _f(m.group(1)) * (1e9 if m.group(2) == "B" else 1e6)
    for t, n in re.findall(r"-\s*type\s+([a-z0-9_]+):\s*([0-9]+)\s*tensors", text, re.I):
        w.tensor_types[t.lower()] = w.tensor_types.get(t.lower(), 0) + int(n)
    m = re.search(r"file type\s*=\s*([^\n]+)", text)
    if m:
        w.file_type = m.group(1).strip()

    # Ollama --verbose blocks
    for blk in re.finditer(r"prompt eval count:\s*" + _num + r"\s*token.*?\n\s*prompt eval duration:\s*([^\n]+)"
                           r"(?:.|\n)*?eval count:\s*" + _num + r"\s*token.*?\n\s*eval duration:\s*([^\n]+)", text):
        p, pd, g, gd = _f(blk.group(1)), _dur(blk.group(2)), _f(blk.group(3)), _dur(blk.group(4))
        w.add(prompt=p, gen=g, prompt_tps=p / pd if pd else None, gen_tps=g / gd if gd else None)

    # llama-server / llama-cli timings (llama.cpp print_timings, llama_perf_context_print)
    pe = list(re.finditer(r"prompt eval time\s*=\s*" + _num + r"\s*ms\s*/\s*" + _num + r"\s*tokens", text))
    ev = list(re.finditer(r"(?<!prompt )\beval time\s*=\s*" + _num + r"\s*ms\s*/\s*" + _num + r"\s*(?:tokens|runs)", text))
    if pe and not w.requests:
        for a, b in zip(pe, ev):
            p, pms, g, gms = _f(a.group(2)), _f(a.group(1)), _f(b.group(2)), _f(b.group(1))
            w.add(prompt=p, gen=g, prompt_tps=p / (pms / 1e3) if pms else None,
                  gen_tps=g / (gms / 1e3) if gms else None)

    # a plain workload line, for readers without a log: prompt_tokens=30000 output_tokens=500
    for a in re.finditer(r"prompt_tokens\s*=\s*" + _num + r"\s+output_tokens\s*=\s*" + _num, text):
        w.add(prompt=_f(a.group(1)), gen=_f(a.group(2)), prompt_tps=None, gen_tps=None)

    # mlx_lm.generate --verbose
    for a in re.finditer(r"Prompt:\s*" + _num + r"\s*tokens,\s*" + _num + r"\s*tokens-per-sec\s*\n\s*Generation:\s*"
                         + _num + r"\s*tokens,\s*" + _num + r"\s*tokens-per-sec", text):
        w.add(prompt=_f(a.group(1)), gen=_f(a.group(3)), prompt_tps=_f(a.group(2)), gen_tps=_f(a.group(4)))

    # timestamps: Ollama GIN access lines, or ISO time= fields
    for d in re.findall(r"\[GIN\]\s*(\d{4}/\d{2}/\d{2}\s*-\s*\d{2}:\d{2}:\d{2})\s*\|\s*200\s*\|[^\n]*POST\s+\"?/api/(?:chat|generate)",
                        text):
        try:
            w.times.append(datetime.strptime(re.sub(r"\s*-\s*", " ", d), "%Y/%m/%d %H:%M:%S"))
        except ValueError:
            pass
    return w


# ----------------------------------------------------------------------------- boxes

# The reference model every calibration run used: gpt-oss-120b (openai/gpt-oss-120b
# config.json), run as the ggml-org MXFP4 GGUF (experts MXFP4, the rest Q8_0) or the
# mlx-community MXFP4-Q8 build (same split). Anchors are the published measurements.
REF_CONFIG = {
    "hidden_size": 2880, "num_hidden_layers": 36, "num_attention_heads": 64, "num_key_value_heads": 8,
    "head_dim": 64, "vocab_size": 201088, "tie_word_embeddings": False, "intermediate_size": 2880,
    "num_local_experts": 128, "num_experts_per_tok": 4, "sliding_window": 128,
    "layer_types": ["sliding_attention", "full_attention"] * 18,
}
REF_BITS = dict(expert=4.25, dense=8.5, head=8.5)

BOXES = {
    "5090": dict(
        name="RTX 5090 32GB", bw=1792, usable=31.0, kind="gpu", price=9500 + 2190,
        price_note="card 9,499.99 from a Newegg marketplace seller, 21 Oct 2026 (absent from NVIDIA's marketplace and Micro Center, Tom's Hardware 17 Oct) + host 2,190 (Ryzen 9 9950X3D 669.99, X870E board 329.99, 64GB DDR5-6000 899.99, 1000W supply 169.99, case 119.99; US listings Oct 2026)",
        load_w=725, idle_w=60, power_note="575W card (NVIDIA) + 150W host, 60W idle: assumed",
        rent=0.69, rent_note="RunPod Community Cloud, RunPod pricing page updated 8 Oct 2026",
        derive="scale_pro6000", sm_ratio=170 / 188),
    "pro6000": dict(
        name="RTX PRO 6000 96GB", bw=1792, usable=94.0, kind="gpu", price=16000 + 2190,
        price_note="card 16,000 on NVIDIA's store since Aug 2026 (Thunder Compute, 16 Oct 2026) + host 2,190 (Ryzen 9 9950X3D 669.99, X870E board 329.99, 64GB DDR5-6000 899.99, 1000W supply 169.99, case 119.99; US listings Oct 2026)",
        load_w=750, idle_w=60, power_note="600W card (NVIDIA) + 150W host, 60W idle: assumed",
        rent=None, rent_note="no hourly rate in the table",
        anchors=dict(engine="llama.cpp",
                     pp_avg={2048: 5518.07, 32768: 4503.36},      # llama.cpp discussion 15396, Aug 2025
                     tg={4096: 221.3, 16384: 193.2, 32768: 178.5})),  # hardware-corner.net, Mar 2026
    "m5max": dict(
        name="Mac Studio M5 Max 128GB", bw=614, usable=96.0, usable_tuned=120.0, kind="mac", price=5399,
        price_note="40-core M5 Max, 128GB, 1TB SSD, Apple Store, Oct 2026",
        load_w=200, idle_w=7, power_note="Apple, 21 Sep 2026 (7W idle, 200W max)",
        rent=None, rent_note="no hourly rate in the table",
        anchors=dict(engine="MLX",
                     pp_avg={16384: 2710.0, 32768: 2537.0},       # hardware-corner.net, Mar 2026
                     tg={4096: 87.9, 16384: 76.0, 32768: 64.5})),
    "m5ultra": dict(
        name="Mac Studio M5 Ultra 256GB", bw=1200, usable=204.8, usable_tuned=248.0, kind="mac", price=9499,
        price_note="M5 Ultra 30-core CPU, 256GB, 1TB SSD, Apple Store 28 Sep 2026",
        load_w=385, idle_w=9, power_note="Apple, 21 Sep 2026 (9W idle, 385W max)",
        rent=None, rent_note="no hourly rate in the table", projected=True,
        derive="scale_m5max", compute_ratio=2.0),
    "spark": dict(
        name="DGX Spark 128GB", bw=273, usable=120.0, kind="spark", price=6950,
        price_note="NVIDIA list since 2 Oct 2026 (The Register); 3,999 at launch in Oct 2025, 4,699 from Feb 2026",
        load_w=240, idle_w=30, power_note="240W adapter rating; 30W idle: assumed",
        rent=0.75, rent_note="Enverge, read 21 Oct 2026",
        anchors=dict(engine="llama.cpp",
                     pp_chunk={0: 2443.91, 32768: 1567.08},        # llama-bench pp2048 @ depth, Feb 2026
                     tg={16: 58.72, 32784: 42.76})),
    "spark64": dict(
        name="DGX Spark 64GB", bw=273, usable=60.0, kind="spark", price=4999,
        price_note="from Acer, ASUS, Dell, Gigabyte, HP and MSI on 23 Oct 2026, starting at 4,999 (NVIDIA blog, 2 Oct 2026); usable memory scaled from the 128GB model's 121GiB",
        load_w=240, idle_w=30, power_note="as the 128GB model: assumed",
        rent=None, rent_note="no hourly rate in the table",
        derive="same_as_spark"),
}
OVERHEAD_GB = 2.7   # llama.cpp compute buffers for gpt-oss-120b at ubatch 2048 (discussion 15396)


def bytes_per_token(model, bits, depth, scale=1.0):
    """Bytes one decode step reads: active experts, every dense weight, the head, the KV cache."""
    w = ((model.active_expert_params + model.shared_params) * bits["expert"]
         + (model.attn_params + model.dense_mlp_params + model.router_params) * bits["dense"]
         + model.head_params * bits["head"]) / 8
    return w * scale + model.kv_read_bytes(depth)


def weights_bytes(model, bits):
    """Bytes the weights occupy in memory, embedding table included."""
    return ((model.expert_params + model.shared_params) * bits["expert"]
            + (model.attn_params + model.dense_mlp_params + model.router_params + model.embed_params) * bits["dense"]
            + (0 if model.tied else model.head_params) * bits["head"]) / 8


def _solve2(a1, b1, c1, a2, b2, c2):
    det = a1 * b2 - a2 * b1
    return (c1 * b2 - c2 * b1) / det, (a1 * c2 - a2 * c1) / det


def calibrate():
    """Per-box decode efficiency and prefill throughputs, from the anchors on gpt-oss-120b."""
    ref = Model(REF_CONFIG, "gpt-oss-120b")
    mm = ref.mm_flops_per_token()
    ad = ref.attn_flops_per_depth()
    cal = {}
    for key, b in BOXES.items():
        a = b.get("anchors")
        if not a:
            continue
        effs = [tps * bytes_per_token(ref, REF_BITS, d) / (b["bw"] * GB) for d, tps in a["tg"].items()]
        if "pp_chunk" in a:           # rate of a 2,048-token chunk that starts at depth d
            (d1, r1), (d2, r2) = sorted(a["pp_chunk"].items())
            tm, ta = _solve2(1, d1 + 1024, 1 / r1, 1, d2 + 1024, 1 / r2)
        else:                         # average rate over a prompt of P tokens from empty
            (p1, r1), (p2, r2) = sorted(a["pp_avg"].items())
            tm, ta = _solve2(p1, p1 * p1 / 2, p1 / r1, p2, p2 * p2 / 2, p2 / r2)
        cal[key] = dict(eff=statistics.mean(effs), fm=mm / tm, fa=ad / ta, engine=a["engine"])
    cal["5090"] = dict(eff=cal["pro6000"]["eff"], fm=cal["pro6000"]["fm"] * BOXES["5090"]["sm_ratio"],
                       fa=cal["pro6000"]["fa"] * BOXES["5090"]["sm_ratio"], engine="llama.cpp")
    cal["m5ultra"] = dict(eff=cal["m5max"]["eff"], fm=cal["m5max"]["fm"] * BOXES["m5ultra"]["compute_ratio"],
                          fa=cal["m5max"]["fa"] * BOXES["m5ultra"]["compute_ratio"], engine="MLX (projected)")
    # The 64GB DGX Spark keeps the 128GB model's chip and 273 GB/s (The Register, 2 Oct 2026), so it
    # takes the 128GB model's calibration and differs only in the memory a model can use.
    cal["spark64"] = dict(cal["spark"])
    # spill anchors on the reference model, RTX 5090 (hardware-corner.net Nov 2025; llama.cpp 15396):
    # a plain layer split ran 3.4 tokens/s on a DDR4 host; experts offloaded ran 8.1 there and
    # about 30 on the guide's host. Each anchor gives the effective GB/s the CPU side delivered.
    pro = cal["pro6000"]
    ref_bpt = bytes_per_token(ref, REF_BITS, 0)
    ref_w = weights_bytes(ref, REF_BITS)
    gpu_rate = BOXES["5090"]["bw"] * GB * pro["eff"]
    plain_cpu = ref_bpt * 19 / 36                                  # 17 of 36 layers stayed on the card
    moe_cpu = ref.active_expert_params * REF_BITS["expert"] / 8 * 21 / 36   # --n-cpu-moe 21
    bw_low = plain_cpu / (1 / 3.4 - (ref_bpt - plain_cpu) / gpu_rate)
    bw_high = moe_cpu / (1 / 30.0 - (ref_bpt - moe_cpu) / gpu_rate)
    cal["spill"] = dict(bw_low=bw_low, bw_high=bw_high, prefill_tps=447.0, prefill_cpu_resident=ref_w * 19 / 36)
    return cal


# ----------------------------------------------------------------------------- predict

def predict(model, bits, scale, prompt, gen, cal, key):
    b = BOXES[key]
    c = cal[key]
    ctx = prompt + gen
    need = weights_bytes(model, bits) * scale + model.kv_read_bytes(ctx) + OVERHEAD_GB * GB
    usable = b["usable"] * GB
    out = dict(key=key, name=b["name"], need_gb=need / GB, usable_gb=b["usable"], projected=b.get("projected", False))
    fits = need <= usable
    tuned = (not fits) and b.get("usable_tuned") and need <= b["usable_tuned"] * GB
    out["fits"] = "yes" if fits else ("after sysctl" if tuned else "no")
    mm = model.mm_flops_per_token()
    ad = model.attn_flops_per_depth()
    if fits or tuned:
        tm = (mm + model.sliding_attn_flops(prompt)) / c["fm"]
        ttft = prompt * tm + (ad / c["fa"]) * prompt * prompt / 2
        bpt = bytes_per_token(model, bits, prompt + gen / 2, scale)
        tps = c["eff"] * b["bw"] * GB / bpt
        out.update(ttft=ttft, tps=tps, tps_range=None, bytes_per_token=bpt, how="fits")
    elif b["kind"] == "gpu":
        s = cal["spill"]
        gpu_rate = c["eff"] * b["bw"] * GB
        w_total = weights_bytes(model, bits) * scale
        gpu_room = max(usable - model.kv_read_bytes(ctx) - OVERHEAD_GB * GB, 1.0)
        frac_cpu = max(0.0, min(1.0, 1 - gpu_room / w_total))
        bpt = bytes_per_token(model, bits, prompt + gen / 2, scale)

        def rate(cpu_bytes, sys_bw):
            return 1.0 / ((bpt - cpu_bytes) / gpu_rate + cpu_bytes / sys_bw)

        plain_cpu = bpt * frac_cpu
        plain = rate(plain_cpu, s["bw_low"])
        exp_w = (model.expert_params + model.shared_params) * bits["expert"] / 8 * scale
        if exp_w and exp_w >= w_total - gpu_room:        # experts alone can leave the card
            moved = (w_total - gpu_room) / exp_w
            cpu_b = model.active_expert_params * bits["expert"] / 8 * scale * moved
            rng, how_rng = (rate(cpu_b, s["bw_low"]), rate(cpu_b, s["bw_high"])), "experts offloaded"
        else:
            rng, how_rng = (plain, rate(plain_cpu, s["bw_high"])), "a faster host"
        pp = s["prefill_tps"] * s["prefill_cpu_resident"] / max(w_total * frac_cpu, 1.0)
        # the anchor scales by bytes left in system memory; the card still does every matmul,
        # so the first token never comes sooner than its own compute allows
        compute = prompt * (mm + model.sliding_attn_flops(prompt)) / c["fm"] + (ad / c["fa"]) * prompt * prompt / 2
        out.update(ttft=max(prompt / pp, compute), tps=plain, tps_range=rng, range_how=how_rng, bytes_per_token=bpt,
                   how="spills", frac_cpu=frac_cpu)
    else:
        out.update(ttft=None, tps=None, tps_range=None, bytes_per_token=None, how="does not fit")
    return out


def yearly(o, gen, per_day, years, kwh):
    b = BOXES[o["key"]]
    if o["tps"] is None:
        return None
    busy_s = o["ttft"] + gen / o["tps"]
    busy_h = per_day * busy_s * 365 / 3600
    busy_h = min(busy_h, 8760)
    energy = (b["load_w"] * busy_h + b["idle_w"] * (8760 - busy_h)) / 1000 * kwh
    own = b["price"] / years + energy
    rent = busy_h * b["rent"] if b["rent"] else None
    return dict(busy_s=busy_s, busy_h=busy_h, own=own, rent=rent, energy=energy)


# ----------------------------------------------------------------------------- report

def fmt_s(x):
    if x is None:
        return "-"
    return "%.0f s" % x if x >= 10 else "%.1f s" % x


def main(argv=None):
    ap = argparse.ArgumentParser(description="Which local-inference box your workload stalls on.")
    ap.add_argument("config", help="the model's config.json")
    ap.add_argument("log", help="Ollama, llama.cpp or mlx_lm log from the machine you run today")
    ap.add_argument("--only", default="", help="comma list of boxes: 5090,pro6000,m5max,m5ultra,spark,spark64")
    ap.add_argument("--quant", default="", help="override: " + ", ".join(PRESETS))
    ap.add_argument("--requests-per-day", type=float, default=None)
    ap.add_argument("--years", type=float, default=3.0, help="years you keep the box (default 3)")
    ap.add_argument("--kwh", type=float, default=0.17, help="electricity price per kWh (default 0.17, assumed)")
    ap.add_argument("--explain", action="store_true", help="print the arithmetic behind every row")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    try:
        with open(a.config, encoding="utf-8") as fh:
            cfg = json.load(fh)
        model = Model(cfg, name=_get(cfg, "_name_or_path", default=a.config))
    except (OSError, ValueError) as e:
        sys.exit("boxfit: cannot read %s as a model's config.json: %s" % (a.config, e))
    if not os.path.isfile(a.log):
        sys.exit("boxfit: no log at %s" % a.log)
    with open(a.log, encoding="utf-8", errors="replace") as fh:
        w = parse_log(fh.read())
    if not w.requests:
        sys.exit("boxfit: no request timings found in the log (Ollama --verbose, llama.cpp timings, or mlx_lm --verbose)")

    # quantization: the log you run beats the config, the config beats the flag, the flag beats the default
    why = None
    if a.quant:
        e, d, h = PRESETS[a.quant]
        bits, why = dict(expert=e, dense=d, head=h), "--quant %s" % a.quant
    elif w.tensor_types and "mxfp4" in w.tensor_types:
        d = 8.5 if "q8_0" in w.tensor_types else 16.0
        bits = dict(expert=4.25, dense=d, head=d)
        why = "log: mxfp4 experts, %s dense" % ("q8_0" if d == 8.5 else "f16")
    elif w.file_type:
        hit = next(((e, d) for pat, e, d in FILE_TYPE_BITS if re.search(pat, w.file_type, re.I)), None)
        if hit:
            bits = dict(expert=hit[0], dense=hit[1], head=hit[1])
            why = "log: file type %s" % w.file_type
    if why is None:
        q = quant_from_config(cfg)
        if q:
            bits, why = q
        else:
            e, d, h = PRESETS["q4_k_m"]
            bits, why = dict(expert=e, dense=d, head=h), "assumed Q4_K_M (no quantization found in the log or config)"
    scale = 1.0
    if w.file_bytes:
        est = weights_bytes(model, bits)
        scale = w.file_bytes / est if est else 1.0

    reqs = w.requests
    prompt = statistics.median(r["prompt"] for r in reqs)
    gen = statistics.median(r["gen"] for r in reqs)
    if a.requests_per_day:
        per_day, per_day_note = a.requests_per_day, "--requests-per-day"
    elif len(w.times) >= 2:
        span = max((max(w.times) - min(w.times)).total_seconds() / 86400, 1.0)
        per_day, per_day_note = len(w.times) / span, "log timestamps"
    else:
        per_day, per_day_note = float(len(reqs)), "assumed the log is one day; set --requests-per-day"

    cal = calibrate()
    keys = [k.strip() for k in a.only.split(",") if k.strip()] or list(BOXES)
    rows = []
    for k in keys:
        if k not in BOXES:
            sys.exit("boxfit: unknown box %r (choose from %s)" % (k, ", ".join(BOXES)))
        o = predict(model, bits, scale, prompt, gen, cal, k)
        o["year"] = yearly(o, gen, per_day, a.years, a.kwh)
        rows.append(o)

    today = [r for r in reqs if r.get("prompt_tps") and r.get("gen_tps")]
    if a.json:
        print(json.dumps(dict(model=dict(total_params=model.total_params, active_params=model.active_params,
                                         bits=bits, scale=scale),
                              workload=dict(prompt=prompt, gen=gen, per_day=per_day), rows=rows), indent=1,
                         default=str))
        return 0

    moe = " MoE %d of %d experts," % (model.k_exp, model.n_exp) if model.n_exp else ""
    print("boxfit %s | %s: %.1fB params, %.1fB active,%s %.2f GB read per token at empty context" % (
        VERSION, model.name, model.total_params / 1e9, model.active_params / 1e9, moe,
        bytes_per_token(model, bits, 0, scale) / GB))
    print("quantization: %s%s" % (why, " (scaled x%.3f to the log's file size)" % scale if abs(scale - 1) > 0.005 else ""))
    print("workload: %d request(s) in the log, median %s prompt tokens and %s generated; %.0f a day (%s)" % (
        len(reqs), "{:,}".format(int(prompt)), "{:,}".format(int(gen)), per_day, per_day_note))
    if today:
        pt = statistics.median(r["prompt_tps"] for r in today)
        gt = statistics.median(r["gen_tps"] for r in today)
        print("your machine today: %s to first token, %.1f tokens per second" % (fmt_s(prompt / pt), gt))
    print()
    print("%-26s %-13s %-13s %-12s %-10s %-10s" % ("box", "fits", "first token", "tokens/s", "own/yr", "rent/yr"))
    for o in rows:
        y = o["year"]
        fits = o["fits"] if o["fits"] != "no" else "no (%.0fGB)" % o["need_gb"]
        if o["how"] == "spills":
            tps = "%.1f (%.0f-%.0f)" % (o["tps"], o["tps_range"][0], o["tps_range"][1]) if o["tps_range"][1] >= 10 else "%.1f (to %.1f)" % (o["tps"], o["tps_range"][1])
        else:
            tps = "%.1f" % o["tps"] if o["tps"] else "-"
        name = o["name"] + (" *" if o["projected"] else "")
        own = "$%s" % "{:,}".format(int(round(y["own"], -1))) if y else "-"
        rent = "$%s" % "{:,}".format(int(round(y["rent"], -1))) if (y and y["rent"] is not None) else "-"
        print("%-26s %-13s %-13s %-12s %-10s %-10s" % (name, fits, fmt_s(o["ttft"]), tps, own, rent))
    print()
    for o in rows:
        if o["how"] == "spills":
            print("does not fit: %s needs %.0fGB and holds %.0f, so it spills to %.1f tokens/s" % (
                o["name"], o["need_gb"], o["usable_gb"], o["tps"]))
            print("  with %s: %.1f to %.1f tokens/s, set by your system RAM and CPU" % ((o["range_how"],) + tuple(o["tps_range"])))
        elif o["how"] == "does not fit":
            print("does not fit: %s needs %.0fGB and holds %.0f" % (o["name"], o["need_gb"], o["usable_gb"]))
    print("fits: weights + KV cache at %s tokens + %.1f GB buffers, against memory the box gives the GPU by default" % (
        "{:,}".format(int(prompt + gen)), OVERHEAD_GB))
    if any(o["how"] == "spills" for o in rows):
        print("spills: plain layer split on the host that ran 3.4 tokens/s, with the range a faster host or --n-cpu-moe reaches")
    if any(o["projected"] for o in rows):
        print("* projected from the M5 Max: the first M5 Ultra reviews (Sep 2026) measured other models, not this calibration model")
    print("own/yr: price over %.0f years + electricity at $%.2f/kWh; rent/yr: the same busy hours at the listed hourly rate" % (
        a.years, a.kwh))
    if a.explain:
        print()
        for o in rows:
            b, c = BOXES[o["key"]], cal[o["key"]]
            print("[%s] %s" % (o["key"], b["name"]))
            print("  memory needed %.1f GB against %.1f GB usable" % (o["need_gb"], o["usable_gb"]))
            if o["how"] == "fits":
                print("  decode: %.2f efficiency x %d GB/s / %.2f GB per token = %.1f tokens/s" % (
                    c["eff"], b["bw"], o["bytes_per_token"] / GB, o["tps"]))
                print("  prefill: %.1f TFLOPS on matmuls, %.1f on attention (%s calibration)" % (
                    c["fm"] / 1e12, c["fa"] / 1e12, c["engine"]))
            if o["year"]:
                y = o["year"]
                print("  busy %.1f s a request, %.0f h a year; energy $%.0f; price $%s (%s)" % (
                    y["busy_s"], y["busy_h"], y["energy"], "{:,}".format(b["price"]), b["price_note"]))
            print("  power: %s; rent: %s" % (b["power_note"], b["rent_note"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
