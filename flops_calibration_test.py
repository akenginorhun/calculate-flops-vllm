
import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from calflops import calculate_flops

# ----------------------------- Helpers -----------------------------

def _measure_prefill_flops(model, tokenizer, P: int, *, print_detailed=False) -> float:
    """Ground-truth prefill FLOPs for a 1xP forward pass."""
    flops, macs, params = calculate_flops(
        model=model,
        input_shape=(1, P),
        transformer_tokenizer=tokenizer,
        forward_mode="forward",
        include_backPropagation=False,
        print_results=False,
        print_detailed=print_detailed,
        output_as_string=False,
    )
    return float(flops)

def _measure_decode_flops(model, tokenizer, P: int, G: int, *, print_detailed=False) -> float:
    """Ground-truth decode FLOPs for generating G tokens from a 1xP prompt."""
    flops, macs, params = calculate_flops(
        model=model,
        input_shape=(1, P),
        transformer_tokenizer=tokenizer,
        forward_mode="generate",
        args=[],
        kwargs={"max_new_tokens": int(G)},
        include_backPropagation=False,
        print_results=False,
        print_detailed=print_detailed,
        output_as_string=False,
    )
    return float(flops)

def _fit_prefill_quadratic(Ps: List[int], y: List[float]) -> Tuple[float, float, float]:
    """Fit c0 + c1*P + c2*P^2 via least squares."""
    X = np.column_stack([np.ones(len(Ps)), np.array(Ps, dtype=float), np.array(Ps, dtype=float) ** 2])
    coeffs, *_ = np.linalg.lstsq(X, np.array(y, dtype=float), rcond=None)
    return tuple(map(float, coeffs))  # c0, c1, c2

def _fit_decode_linear(Ps: List[int], Gs: List[int], y: List[float]) -> Tuple[float, float]:
    """
    Fit d0*G + d1*(P*G + G*(G-1)/2) via least squares.
    Construct features:
      f1 = G
      f2 = P*G + G*(G-1)/2
    """
    assert len(Ps) == len(Gs) == len(y)
    Gs_arr = np.array(Gs, dtype=float)
    Ps_arr = np.array(Ps, dtype=float)
    f1 = Gs_arr
    f2 = Ps_arr * Gs_arr + (Gs_arr * (Gs_arr - 1.0) / 2.0)
    X = np.column_stack([f1, f2])
    coeffs, *_ = np.linalg.lstsq(X, np.array(y, dtype=float), rcond=None)
    d0, d1 = coeffs
    return float(d0), float(d1)

def _prefill_predict(P: int, c0: float, c1: float, c2: float) -> float:
    return c0 + c1 * P + c2 * (P ** 2)

def _decode_predict(P: int, G: int, d0: float, d1: float) -> float:
    return d0 * G + d1 * (P * G + G * (G - 1) / 2.0)

def _percent_err(y_true: float, y_pred: float) -> float:
    if y_true == 0:
        return 0.0 if y_pred == 0 else float("inf")
    return abs(y_pred - y_true) / abs(y_true) * 100.0

@dataclass
class Regime:
    name: str
    prefill_points: int
    decode_points: int

# ----------------------------- Main routine -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="HF model id for AutoModelForCausalLM")
    parser.add_argument("--access_token", type=str, default="", help="HF access token if needed")
    parser.add_argument("--trust_remote_code", action="store_true", help="Pass trust_remote_code=True")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefill_points", type=int, nargs="+", default=[3, 6], help="List of #prefill calibration points per regime")
    parser.add_argument("--decode_points", type=int, nargs="+", default=[2, 4], help="List of #decode calibration points per regime")
    parser.add_argument("--test_grid_P", type=int, nargs="+", default=[96, 192, 384, 1536, 3072], help="Held-out P values")
    parser.add_argument("--test_grid_G", type=int, nargs="+", default=[1, 16, 64, 256], help="Held-out G values")
    parser.add_argument("--min_P", type=int, default=64)
    parser.add_argument("--max_P", type=int, default=4096)
    parser.add_argument("--min_G", type=int, default=1)
    parser.add_argument("--max_G", type=int, default=512)
    parser.add_argument("--print_detailed", action="store_true", help="Ask calflops to print detailed per-module table once per call")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load model & tokenizer to CPU once
    print(f"[info] Loading {args.model_name} on {args.device} ...")
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code, token=(args.access_token or None))
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        trust_remote_code=args.trust_remote_code,
        token=(args.access_token or None),
        torch_dtype=torch.float32,
        device_map=None
    )
    model.to(args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=(args.access_token or None), trust_remote_code=args.trust_remote_code)

    # Build two calibration regimes
    regimes: List[Regime] = []
    plen = len(args.prefill_points)
    dlen = len(args.decode_points)
    if plen != dlen:
        # Pair them up to the min length
        n = min(plen, dlen)
        prefill_list = args.prefill_points[:n]
        decode_list = args.decode_points[:n]
    else:
        n = plen
        prefill_list = args.prefill_points
        decode_list = args.decode_points

    for i in range(n):
        regimes.append(Regime(name=f"R{i+1}_P{prefill_list[i]}_G{decode_list[i]}", prefill_points=prefill_list[i], decode_points=decode_list[i]))

    # Function to pick calibration points
    def pick_prefill_points(k: int) -> List[int]:
        # Log-spaced between min_P and max_P, rounded to nearest 8
        Ps = np.unique((np.geomspace(args.min_P, args.max_P, num=k)).astype(int)).tolist()
        return Ps

    def pick_decode_points(k: int) -> List[Tuple[int, int]]:
        # Mix of varying P and G
        Ps = np.unique((np.geomspace(args.min_P, args.max_P, num=max(2, min(4, k)))).astype(int)).tolist()
        Gs = np.unique((np.geomspace(args.min_G, args.max_G, num=k)).astype(int)).tolist()
        pairs = []
        # Interleave to get k pairs
        idxP = 0
        for g in Gs:
            p = Ps[idxP % len(Ps)]
            pairs.append((p, g))
            idxP += 1
            if len(pairs) >= k:
                break
        return pairs

    # Prepare held-out grid (ground truth evaluation)
    test_pairs = [(int(P), int(G)) for P in args.test_grid_P for G in args.test_grid_G]

    # Pre-measure ground truth for all needed points (calibration + test)
    # Cache ground-truth calls to avoid re-running calflops for duplicates
    gt_prefill_cache: Dict[int, float] = {}
    gt_decode_cache: Dict[Tuple[int, int], float] = {}

    def gt_prefill(P: int) -> float:
        if P not in gt_prefill_cache:
            gt_prefill_cache[P] = _measure_prefill_flops(model, tokenizer, P, print_detailed=args.print_detailed)
        return gt_prefill_cache[P]

    def gt_decode(P: int, G: int) -> float:
        key = (P, G)
        if key not in gt_decode_cache:
            gt_decode_cache[key] = _measure_decode_flops(model, tokenizer, P, G, print_detailed=args.print_detailed)
        return gt_decode_cache[key]

    # Also include calibration points in GT cache as we go
    all_results = []

    for reg in regimes:
        print(f"\n[regime] {reg.name}: calibrating with {reg.prefill_points} prefill and {reg.decode_points} decode points")

        # Calibration point selection
        cal_Ps = pick_prefill_points(reg.prefill_points)
        cal_Ds = pick_decode_points(reg.decode_points)

        # Measure GT for calibration points
        cal_prefill_y = [gt_prefill(P) for P in cal_Ps]
        cal_decode_Ps = [p for (p, g) in cal_Ds]
        cal_decode_Gs = [g for (p, g) in cal_Ds]
        cal_decode_y = [gt_decode(p, g) for (p, g) in cal_Ds]

        # Fit coefficients
        c0, c1, c2 = _fit_prefill_quadratic(cal_Ps, cal_prefill_y)
        d0, d1 = _fit_decode_linear(cal_decode_Ps, cal_decode_Gs, cal_decode_y)

        # Evaluate on held-out grid
        rows = []
        perfill_errs = []
        decode_errs = []
        total_errs = []

        for (P, G) in test_pairs:
            y_prefill = gt_prefill(P)
            y_decode = gt_decode(P, G)
            y_total = y_prefill + y_decode

            y_prefill_pred = _prefill_predict(P, c0, c1, c2)
            y_decode_pred = _decode_predict(P, G, d0, d1)
            y_total_pred = y_prefill_pred + y_decode_pred

            e_prefill = _percent_err(y_prefill, y_prefill_pred)
            e_decode = _percent_err(y_decode, y_decode_pred)
            e_total = _percent_err(y_total, y_total_pred)

            perfill_errs.append(e_prefill)
            decode_errs.append(e_decode)
            total_errs.append(e_total)

            rows.append({
                "P": P, "G": G,
                "gt_prefill": y_prefill, "gt_decode": y_decode, "gt_total": y_total,
                "pred_prefill": y_prefill_pred, "pred_decode": y_decode_pred, "pred_total": y_total_pred,
                "err_prefill_%": e_prefill, "err_decode_%": e_decode, "err_total_%": e_total
            })

        def summarize(errs: List[float]) -> Dict[str, float]:
            arr = np.array(errs, dtype=float)
            return {
                "MAPE%": float(np.mean(arr)),
                "MedAPE%": float(np.median(arr)),
                "MaxAPE%": float(np.max(arr)),
                "P90APE%": float(np.percentile(arr, 90)),
            }

        summary = {
            "regime": reg.name,
            "prefill_points": reg.prefill_points,
            "decode_points": reg.decode_points,
            "prefill_err": summarize(perfill_errs),
            "decode_err": summarize(decode_errs),
            "total_err": summarize(total_errs),
        }
        all_results.append({"summary": summary, "rows": rows})

        # Print summary nicely
        print(f"[summary] {reg.name}")
        for k in ["prefill_err", "decode_err", "total_err"]:
            s = summary[k]
            print(f"  {k}: MAPE={s['MAPE%']:.2f}%  Med={s['MedAPE%']:.2f}%  P90={s['P90APE%']:.2f}%  Max={s['MaxAPE%']:.2f}%")

    # Save results
    out_json = "flops_calibration_results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[done] Wrote detailed results to {out_json}")

if __name__ == "__main__":
    main()
