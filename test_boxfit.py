#!/usr/bin/env python3
"""Checks boxfit against numbers it was not allowed to invent. Run: python3 test_boxfit.py"""
import io
import json
import os
import unittest
from contextlib import redirect_stdout

import boxfit as bf

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.join(HERE, "examples")


def cfg(name):
    with open(os.path.join(EX, name), encoding="utf-8") as fh:
        return json.load(fh)


class ModelCounts(unittest.TestCase):
    def test_gpt_oss_120b_params_match_llama_cpp(self):
        m = bf.Model(cfg("gpt-oss-120b.config.json"))
        # llama.cpp reports 116.83 B for this model; OpenAI's model card says 5.1B active
        self.assertAlmostEqual(m.total_params / 1e9, 116.83, delta=116.83 * 0.01)
        self.assertAlmostEqual(m.active_params / 1e9, 5.1, delta=5.1 * 0.03)

    def test_gpt_oss_bytes_per_token_match_gguf_header(self):
        m = bf.Model(cfg("gpt-oss-120b.config.json"))
        # the ggml-org MXFP4 GGUF header gives 1.904 GB of active experts + 1.686 GB dense = 3.59 GB
        self.assertAlmostEqual(bf.bytes_per_token(m, bf.REF_BITS, 0) / 1e9, 3.59, delta=3.59 * 0.02)

    def test_gpt_oss_weights_match_file_size(self):
        m = bf.Model(cfg("gpt-oss-120b.config.json"))
        # the GGUF is 63.37 GB (59.02 GiB)
        self.assertAlmostEqual(bf.weights_bytes(m, bf.REF_BITS) / 1e9, 63.37, delta=63.37 * 0.02)

    def test_qwen36_is_35b_with_3b_active(self):
        m = bf.Model(cfg("qwen3.6-35b-a3b-nvfp4.config.json"))
        self.assertAlmostEqual(m.total_params / 1e9, 35.0, delta=1.0)
        self.assertTrue(2.7e9 < m.active_params < 3.2e9, m.active_params)

    def test_llama_33_70b_dense_count(self):
        m = bf.Model(cfg("llama-3.3-70b.config.json"))
        self.assertAlmostEqual(m.total_params / 1e9, 70.6, delta=0.7)
        self.assertEqual(m.n_exp, 0)

    def test_qwen3_30b_a3b_counts(self):
        m = bf.Model(cfg("qwen3-30b-a3b.config.json"))
        self.assertAlmostEqual(m.total_params / 1e9, 30.5, delta=0.6)
        self.assertAlmostEqual(m.active_params / 1e9, 3.3, delta=0.3)


class Calibration(unittest.TestCase):
    def setUp(self):
        self.cal = bf.calibrate()
        self.ref = bf.Model(bf.REF_CONFIG)

    def tps(self, key, depth):
        b, c = bf.BOXES[key], self.cal[key]
        return c["eff"] * b["bw"] * bf.GB / bf.bytes_per_token(self.ref, bf.REF_BITS, depth)

    def test_spark_reproduces_llama_cpp(self):
        self.assertAlmostEqual(self.tps("spark", 16), 58.72, delta=58.72 * 0.05)
        self.assertAlmostEqual(self.tps("spark", 32784), 42.76, delta=42.76 * 0.05)

    def test_m5max_reproduces_mlx(self):
        self.assertAlmostEqual(self.tps("m5max", 32768), 64.5, delta=64.5 * 0.10)

    def test_pro6000_reproduces_32k(self):
        self.assertAlmostEqual(self.tps("pro6000", 32768), 178.5, delta=178.5 * 0.10)

    def test_spark_32k_prefill_about_17_seconds(self):
        m = self.ref
        c = self.cal["spark"]
        p = 32768
        ttft = p * (m.mm_flops_per_token() + m.sliding_attn_flops(p)) / c["fm"] + m.attn_flops_per_depth() / c["fa"] * p * p / 2
        self.assertAlmostEqual(ttft, 16.9, delta=1.0)


class Predictions(unittest.TestCase):
    def setUp(self):
        self.cal = bf.calibrate()

    def test_gpt_oss_30k_fits_everything_but_the_5090(self):
        m = bf.Model(cfg("gpt-oss-120b.config.json"))
        fits = {k: bf.predict(m, bf.REF_BITS, 1.0, 30000, 500, self.cal, k)["fits"] for k in bf.BOXES}
        self.assertEqual(fits["5090"], "no")
        for k in ("pro6000", "m5max", "m5ultra", "spark"):
            self.assertEqual(fits[k], "yes", k)

    def test_known_case_qwen36_on_one_spark(self):
        """NVIDIA measured 101.8 tokens per second (August 2026 Reviewer's Guide, 2k context,
        three-token MTP speculation). boxfit is calibrated on gpt-oss-120b in llama.cpp and does
        not model speculation, so it must land below the measurement and within 15 percent."""
        c = cfg("qwen3.6-35b-a3b-nvfp4.config.json")
        m = bf.Model(c)
        bits, _ = bf.quant_from_config(c)
        o = bf.predict(m, bits, 1.0, 2048, 256, self.cal, "spark")
        self.assertLess(o["tps"], 101.8)
        self.assertGreater(o["tps"], 101.8 * 0.85)


    def test_a_spilling_card_is_never_faster_to_first_token_than_its_own_compute(self):
        """A dense 70B at Q4_K_M spills off the 32GB RTX 5090. Weights in system memory only add
        time: the card still does every matrix multiply, so its first token can't come sooner
        than the same card's compute allows, nor sooner than the 96GB PRO 6000 that holds it."""
        m = bf.Model(cfg("llama-3.3-70b.config.json"))
        e, d, h = bf.PRESETS["q4_k_m"]
        bits = dict(expert=e, dense=d, head=h)
        spill = bf.predict(m, bits, 1.0, 2048, 256, self.cal, "5090")
        fit = bf.predict(m, bits, 1.0, 2048, 256, self.cal, "pro6000")
        self.assertEqual(spill["how"], "spills")
        c = self.cal["5090"]
        compute = 2048 * m.mm_flops_per_token() / c["fm"] + (m.attn_flops_per_depth() / c["fa"]) * 2048 * 2048 / 2
        self.assertGreaterEqual(spill["ttft"], compute)
        self.assertGreaterEqual(spill["ttft"], fit["ttft"])

class LogParsing(unittest.TestCase):
    # Synthetic strings in each tool's documented output shape, not captured runs.
    OLLAMA = """total duration:       14.2s
load duration:        40ms
prompt eval count:    2048 token(s)
prompt eval duration: 1.6s
prompt eval rate:     1280.00 tokens/s
eval count:           256 token(s)
eval duration:        6.4s
eval rate:            40.00 tokens/s
"""
    SERVER = """print_info: file type   = Q4_K - Medium
print_info: file size   = 18.63 GiB (4.89 BPW)
prompt eval time =    2000.00 ms /  4096 tokens (    0.49 ms per token,  2048.00 tokens per second)
       eval time =    8000.00 ms /   400 tokens (   20.00 ms per token,    50.00 tokens per second)
"""
    MLX = """Prompt: 4096 tokens, 1325.062 tokens-per-sec
Generation: 128 tokens, 87.873 tokens-per-sec
Peak memory: 64.408 GB
"""
    GIN = "\n".join('[GIN] 2026/09/2%d - 1%d:00:00 | 200 |  12.0s | 127.0.0.1 | POST "/api/chat"' % (d, h)
                    for d in (6, 7) for h in range(4))

    def test_ollama_verbose(self):
        w = bf.parse_log(self.OLLAMA)
        r = w.requests[0]
        self.assertEqual((r["prompt"], r["gen"]), (2048, 256))
        self.assertAlmostEqual(r["prompt_tps"], 1280, delta=1)
        self.assertAlmostEqual(r["gen_tps"], 40, delta=0.1)

    def test_llama_server(self):
        w = bf.parse_log(self.SERVER)
        self.assertAlmostEqual(w.file_bytes / 2 ** 30, 18.63, places=2)
        self.assertEqual(w.bpw, 4.89)
        r = w.requests[0]
        self.assertEqual((r["prompt"], r["gen"]), (4096, 400))
        self.assertAlmostEqual(r["gen_tps"], 50, delta=0.01)

    def test_mlx_lm(self):
        r = bf.parse_log(self.MLX).requests[0]
        self.assertEqual((r["prompt"], r["gen"], r["gen_tps"]), (4096, 128, 87.873))

    def test_ollama_gin_timestamps(self):
        w = bf.parse_log(self.GIN)
        self.assertEqual(len(w.times), 8)

    def test_loader_tensor_types(self):
        with open(os.path.join(EX, "gpt-oss-120b-agent.log"), encoding="utf-8") as fh:
            w = bf.parse_log(fh.read())
        self.assertEqual(w.tensor_types, {"f32": 433, "q8_0": 146, "mxfp4": 108})
        self.assertEqual(len(w.requests), 2)


class PrefixCache(unittest.TestCase):
    """A warm turn reads only its new tokens but carries the whole context (2026-10-24)."""

    def setUp(self):
        self.m = bf.Model(cfg("gpt-oss-120b.config.json"))
        self.cal = bf.calibrate()

    def run_(self, prompt, cached, key="spark"):
        return bf.predict(self.m, bf.REF_BITS, 1.0, prompt, 500, self.cal, key, cached)

    def test_warm_turn_reads_only_new_tokens(self):
        cold, warm = self.run_(30000, 0), self.run_(1200, 28800)
        self.assertLess(warm["ttft"], cold["ttft"] / 10)

    def test_warm_turn_carries_the_whole_context(self):
        cold, warm, short = self.run_(30000, 0), self.run_(1200, 28800), self.run_(1200, 0)
        self.assertAlmostEqual(warm["need_gb"], cold["need_gb"], places=6)
        self.assertAlmostEqual(warm["tps"], cold["tps"], places=6)
        self.assertLess(warm["tps"], short["tps"])          # more cache to read every step
        self.assertGreater(warm["need_gb"], short["need_gb"])

    def test_workload_line_carries_cached_tokens(self):
        w = bf.parse_log("cached_tokens=28800 prompt_tokens=1200 output_tokens=500\nprompt_tokens=30000 output_tokens=500\n")
        self.assertEqual([r.get("cached") for r in w.requests], [28800.0, 0.0])

    def test_session_example_runs(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = bf.main([os.path.join(EX, "gpt-oss-120b.config.json"), os.path.join(EX, "gpt-oss-120b-session.log"),
                          "--requests-per-day", "200", "--only", "spark"])
        self.assertEqual(rc, 0)
        self.assertIn("prefix cache: 28,800 tokens", buf.getvalue())


class SpillLeadsWithOffload(unittest.TestCase):
    """A MoE model that spills leads with the experts-offloaded range; the plain split is only a
    line not to run (2026-10-26)."""

    def test_moe_spill_reports_offload_range(self):
        m = bf.Model(cfg("gpt-oss-120b.config.json"))
        o = bf.predict(m, bf.REF_BITS, 1.0, 1200, 500, bf.calibrate(), "5090", 28800)
        self.assertEqual(o["how"], "spills")
        self.assertGreater(o["tps"], o["plain"])
        self.assertLess(o["tps"], o["tps_hi"])

    def test_cli_names_the_plain_split_as_the_line_not_to_run(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            bf.main([os.path.join(EX, "gpt-oss-120b.config.json"), os.path.join(EX, "gpt-oss-120b-session.log"),
                     "--requests-per-day", "200", "--only", "5090"])
        out = buf.getvalue()
        self.assertIn("the experts go to system memory", out)
        self.assertIn("do not run a plain layer split", out)


class EndToEnd(unittest.TestCase):
    def test_cli_runs_and_names_every_box(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = bf.main([os.path.join(EX, "gpt-oss-120b.config.json"), os.path.join(EX, "gpt-oss-120b-agent.log"),
                          "--requests-per-day", "200"])
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        for name in ("RTX 5090", "RTX PRO 6000", "M5 Max", "M5 Ultra", "DGX Spark"):
            self.assertIn(name, out)


if __name__ == "__main__":
    unittest.main(verbosity=1)
