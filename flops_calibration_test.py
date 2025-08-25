
import argparse
import numpy as np
import torch
from calflops import calculate_flops
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


def load_model(model_name):
    """Loads a model and tokenizer from Hugging Face, ensuring it's on the CPU."""
    print(f"Loading model and tokenizer for '{model_name}'...")
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model.to("cpu")
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer


def get_ground_truth_flops(model, tokenizer, prompt_len, gen_len):
    """
    Measures the actual FLOPs for prefill and decode using calflops.
    """
    if prompt_len == 0 and gen_len == 0:
        return 0, 0

    # Prepare input for the model
    input_ids = torch.ones(1, prompt_len, dtype=torch.long)
    
    # 1. Calculate total FLOPs for generate (prefill + decode)
    total_flops, _, _ = calculate_flops(
        model=model,
        kwargs={"input_ids": input_ids, "max_new_tokens": gen_len},
        forward_mode="generate",
        output_as_string=False,
        print_results=False,
    )

    # 2. Calculate FLOPs for prefill only
    if prompt_len > 0:
        prefill_flops, _, _ = calculate_flops(
            model=model,
            kwargs={"input_ids": input_ids},
            forward_mode="forward",
            output_as_string=False,
            print_results=False,
        )
    else:
        prefill_flops = 0

    # 3. Decode FLOPs is the difference
    decode_flops = total_flops - prefill_flops
    
    return prefill_flops, decode_flops


def fit_prefill(prompt_lengths, prefill_flops):
    """Fits a quadratic model to the prefill FLOPs."""
    if len(prompt_lengths) < 3:
        raise ValueError("Need at least 3 points to fit a quadratic model for prefill.")
    coeffs = np.polyfit(prompt_lengths, prefill_flops, 2)
    return coeffs


def predict_prefill(prompt_len, coeffs):
    """Predicts prefill FLOPs using the fitted quadratic model."""
    return np.polyval(coeffs, prompt_len)


def fit_decode(prompt_lengths, gen_lengths, decode_flops):
    """
    Fits a linear series model to the decode FLOPs.
    Model: FLOPs = a * (L*S + L*(L-1)/2) + b*L
    """
    if len(prompt_lengths) < 2:
        raise ValueError("Need at least 2 points to fit the decode model.")
        
    X1 = np.array([L*S + L*(L-1)/2 for S, L in zip(prompt_lengths, gen_lengths)])
    X2 = np.array(gen_lengths)
    A = np.vstack([X1, X2]).T
    
    coeffs, _, _, _ = np.linalg.lstsq(A, decode_flops, rcond=None)
    return coeffs


def predict_decode(prompt_len, gen_len, coeffs):
    """Predicts decode FLOPs using the fitted linear series model."""
    a, b = coeffs
    x1 = gen_len * prompt_len + gen_len * (gen_len - 1) / 2
    x2 = gen_len
    return a * x1 + b * x2
