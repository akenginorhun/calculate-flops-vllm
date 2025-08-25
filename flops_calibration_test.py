import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from calflops import calculate_flops
import random

def load_model(model_name):
    """
    Loads a model and tokenizer from Hugging Face onto the CPU.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        tuple: A tuple containing the loaded model and tokenizer.
    """
    print(f"Loading model: {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Model loaded successfully.")
    return model, tokenizer

def get_ground_truth_flops(model, tokenizer, prompt_len, gen_len):
    """
    Measures the ground-truth FLOPs for prefill and decode using calflops.

    Args:
        model (torch.nn.Module): The model to measure.
        tokenizer: The tokenizer for the model.
        prompt_len (int): The length of the prompt.
        gen_len (int): The length of the generated sequence.

    Returns:
        tuple: A tuple containing the prefill and per-token decode FLOPs as floats.
    """
    # Prefill
    inputs = tokenizer("a" * prompt_len, return_tensors="pt")
    # Set output_as_string=False to get a direct numerical output
    prefill_flops, _, _ = calculate_flops(model=model, 
                                          kwargs=inputs, 
                                          print_results=False, 
                                          output_as_string=False)

    # Decode
    # To measure per-token decode FLOPs, we measure total FLOPs for N and N+1 tokens
    # and take the difference.
    if gen_len > 0:
        inputs_n = tokenizer("a" * (prompt_len + gen_len - 1), return_tensors="pt")
        flops_n, _, _ = calculate_flops(model=model, 
                                        kwargs=inputs_n, 
                                        print_results=False, 
                                        output_as_string=False)

        inputs_n_plus_1 = tokenizer("a" * (prompt_len + gen_len), return_tensors="pt")
        flops_n_plus_1, _, _ = calculate_flops(model=model, 
                                               kwargs=inputs_n_plus_1, 
                                               print_results=False, 
                                               output_as_string=False)
        
        decode_flops = flops_n_plus_1 - flops_n
    else:
        decode_flops = 0.0

    return prefill_flops, decode_flops

def calibrate(model, tokenizer, prefill_points, decode_points):
    """
    Fits a simple formula to measured FLOPs points.

    Args:
        model (torch.nn.Module): The model to calibrate.
        tokenizer: The tokenizer for the model.
        prefill_points (list): A list of prompt lengths for prefill calibration.
        decode_points (list): A list of generation lengths for decode calibration.

    Returns:
        tuple: A tuple containing the fitted prefill (quadratic) and decode (linear) functions.
    """
    print("Running calibration...")
    # Prefill calibration
    prefill_x = np.array(prefill_points)
    prefill_y = np.array([get_ground_truth_flops(model, tokenizer, p, 1)[0] for p in prefill_points])
    # Fit a quadratic: y = ax^2 + bx + c
    prefill_coeffs = np.polyfit(prefill_x, prefill_y, 2)
    prefill_func = np.poly1d(prefill_coeffs)

    # Decode calibration (fitting a constant value, as per-token FLOPs should be constant)
    decode_y = np.array([get_ground_truth_flops(model, tokenizer, 1, d)[1] for d in decode_points])
    # Fit a constant (degree 0 polynomial) which is just the average
    decode_flop_per_token = np.mean(decode_y)
    
    print("Calibration complete.")
    return prefill_func, decode_flop_per_token

def predict_flops(prefill_func, decode_flop_per_token, prompt_len, gen_len):
    """
    Predicts FLOPs using the calibrated formulas.

    Args:
        prefill_func (np.poly1d): The fitted prefill function.
        decode_flop_per_token (float): The calibrated per-token decode FLOPs.
        prompt_len (int): The prompt length.
        gen_len (int): The generation length.

    Returns:
        tuple: A tuple containing the predicted prefill and total decode FLOPs.
    """
    prefill_flops = prefill_func(prompt_len)
    # Total decode FLOPs is per-token FLOPs multiplied by the number of tokens
    total_decode_flops = decode_flop_per_token * gen_len
    return prefill_flops, total_decode_flops


def run_test(model, tokenizer, prefill_func, decode_flop_per_token, test_grid):
    """
    Compares predictions from calibration against ground truth.

    Args:
        model (torch.nn.Module): The model to test.
        tokenizer: The tokenizer for the model.
        prefill_func (np.poly1d): The fitted prefill function.
        decode_flop_per_token (float): The calibrated per-token decode FLOPs.
        test_grid (list): A list of (prompt_len, gen_len) tuples for testing.
    """
    print("\nRunning tests against ground truth...")
    results = []
    for prompt_len, gen_len in test_grid:
        gt_prefill, gt_decode_per_token = get_ground_truth_flops(model, tokenizer, prompt_len, gen_len)
        gt_total_decode = gt_decode_per_token * gen_len

        pred_prefill, pred_total_decode = predict_flops(prefill_func, decode_flop_per_token, prompt_len, gen_len)

        prefill_error = abs(pred_prefill - gt_prefill) / gt_prefill * 100 if gt_prefill > 0 else 0
        decode_error = abs(pred_total_decode - gt_total_decode) / gt_total_decode * 100 if gt_total_decode > 0 else 0

        results.append({
            "prompt_len": prompt_len,
            "gen_len": gen_len,
            "gt_prefill": gt_prefill,
            "pred_prefill": pred_prefill,
            "prefill_error": prefill_error,
            "gt_decode": gt_total_decode,
            "pred_decode": pred_total_decode,
            "decode_error": decode_error,
        })

    return results

def report_results(results):
    """
    Reports the accuracy and results of the calibration test.

    Args:
        results (list): A list of dictionaries containing the test results.
    """
    print("\n--- Test Results ---")
    avg_prefill_error = np.mean([r["prefill_error"] for r in results])
    max_prefill_error = np.max([r["prefill_error"] for r in results])
    avg_decode_error = np.mean([r["decode_error"] for r in results])
    max_decode_error = np.max([r["decode_error"] for r in results])

    print(f"Average Prefill Error: {avg_prefill_error:.2f}%")
    print(f"Maximum Prefill Error: {max_prefill_error:.2f}%")
    print(f"Average Decode Error: {avg_decode_error:.2f}%")
    print(f"Maximum Decode Error: {max_decode_error:.2f}%")

    print("\n--- Detailed Results (FLOPs) ---")
    print(f"{'Prompt':>8} | {'Gen':>5} | {'GT Prefill':>12} | {'Pred Prefill':>12} | {'Error (%)':>10} | {'GT Decode':>12} | {'Pred Decode':>12} | {'Error (%)':>10}")
    print("-" * 105)
    for r in results:
        print(
            f"{r['prompt_len']:>8} | {r['gen_len']:>5} | "
            f"{r['gt_prefill']:>12.2e} | {r['pred_prefill']:>12.2e} | {r['prefill_error']:>10.2f} | "
            f"{r['gt_decode']:>12.2e} | {r['pred_decode']:>12.2e} | {r['decode_error']:>10.2f}"
        )


def main():
    """
    Main function to run the one-time calibration test.
    Splits a dataset into training and testing sets, then tests
    multiple calibration setups by sampling from the training set.
    """
    # --- Parameters to Change ---
    model_name = "sshleifer/tiny-gpt2"

    # Define the number of points to use for calibration in each test run.
    # Each tuple means (num_points_for_prefill_fit, num_points_for_decode_fit)
    calibration_configs = [
        (3, 2),  # Use 3 points to fit prefill, 2 to fit decode
        (6, 4),
        (12, 8)
    ]

    # This is now our master dataset of possible prompt/generation lengths.
    full_dataset = [
        (10, 20), (32, 32), (50, 200), (73, 156), (92, 1078),
        (128, 64), (157, 423), (200, 50), (214, 332), (256, 256),
        (341, 89), (400, 1283), (418, 1234), (500, 1), (5, 512),
        (567, 234), (789, 45), (1000, 1000)
    ]
    random.shuffle(full_dataset) # Shuffle to ensure random splits

    # Split the dataset into a training pool (for calibration) and a testing set.
    # Let's use ~30% for training and the rest for testing.
    training_size = int(len(full_dataset) * 0.3)
    training_data = full_dataset[:training_size]
    test_data = full_dataset[training_size:]

    print(f"Total dataset size: {len(full_dataset)}")
    print(f"Using {len(training_data)} points for calibration pool and {len(test_data)} for testing.")

    # --- Script Execution ---
    model, tokenizer = load_model(model_name)

    for i, (num_prefill, num_decode) in enumerate(calibration_configs):
        print(f"\n{'='*60}")
        print(f"--- Running Test Setup {i+1}: Calibrating with {num_prefill} prefill and {num_decode} decode points ---")
        print(f"{'='*60}")

        # Ensure we don't request more samples than are available in the training pool.
        if num_prefill > len(training_data) or num_decode > len(training_data):
            print(f"Skipping setup. Not enough data in training pool for this configuration.")
            continue

        # Randomly sample points from our training data for this specific calibration run.
        prefill_samples = random.sample(training_data, num_prefill)
        decode_samples = random.sample(training_data, num_decode)

        # Extract just the prompt lengths for prefill, and generation lengths for decode.
        prefill_cal_points = [p[0] for p in prefill_samples]
        decode_cal_points = [p[1] for p in decode_samples]
        
        print(f"Prefill calibration prompt lengths: {sorted(prefill_cal_points)}")
        print(f"Decode calibration generation lengths: {sorted(decode_cal_points)}")

        # Calibrate using the sampled training points.
        prefill_func, decode_flop_per_token = calibrate(model, tokenizer, prefill_cal_points, decode_cal_points)
        
        # Run the test against the unseen test data.
        results = run_test(model, tokenizer, prefill_func, decode_flop_per_token, test_data)
        
        # Report the results for this setup.
        report_results(results)


if __name__ == "__main__":
    main()