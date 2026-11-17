"""
Unified Benchmark Execution Script.
Executes all scientific evaluations and prints structured benchmark tables.
"""

import os
import sys
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.ablation_experiments import (
    run_quantization_bitrate_ablation,
    run_telephony_robustness_ablation,
    run_latency_budget_profiling
)


def run_all_benchmarks():
    print("=" * 70)
    print(" BLAND AUDIO RESEARCH: SYSTEM BENCHMARKS & EXPERIMENTAL ABLATIONS ")
    print("=" * 70)

    # 1. Latency budget
    print("\n[1] LATENCY BUDGET & REAL-TIME PROFILING:")
    budget = run_latency_budget_profiling()
    for k, v in budget.items():
        print(f"  - {k:<40}: {v}")

    # 2. Quantization bitrate ablation
    print("\n[2] NEURAL CODEC BITRATE vs RECONSTRUCTION FIDELITY (MCD & SI-SDR):")
    bitrate_results = run_quantization_bitrate_ablation()
    print(f"  {'Configuration':<35} | {'Bitrate':<8} | {'MCD (dB)':<8} | {'SI-SDR (dB)':<11} | {'Spectral Conv':<13}")
    print("  " + "-" * 85)
    for r in bitrate_results:
        print(f"  {r['configuration']:<35} | {r['bitrate_kbps']:<8} | {r['mcd_db']:<8} | {r['si_sdr_db']:<11} | {r['spectral_convergence']:<13}")

    # 3. Telephony robustness ablation
    print("\n[3] TELEPHONY CHANNEL ROBUSTNESS & PACKET LOSS CONCEALMENT (PLC):")
    telephony_results = run_telephony_robustness_ablation()
    print(f"  {'Channel Condition':<40} | {'MCD (dB)':<8} | {'SI-SDR (dB)':<11} | {'Perceptual Status'}")
    print("  " + "-" * 90)
    for r in telephony_results:
        print(f"  {r['channel_condition']:<40} | {r['mcd_db']:<8} | {r['si_sdr_db']:<11} | {r['perceptual_status']}")

    print("\n" + "=" * 70)
    print(" ALL BENCHMARKS COMPLETED SUCCESSFULLY.")
    print("=" * 70)


if __name__ == "__main__":
    run_all_benchmarks()
