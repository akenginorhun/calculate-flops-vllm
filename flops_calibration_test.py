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
    Uses a more robust method for decode FLOPs calculation.
    """
    # ... (the safety checks for max_len can remain if you have them) ...
    
    # Prefill
    inputs = tokenizer("a" * prompt_len, return_tensors="pt")
    prefill_flops, _, _ = calculate_flops(model=model, kwargs=inputs, print_results=False, output_as_string=False)

    # Decode
    if gen_len > 0:
        # --- ROBUST DECODE LOGIC ---
        # Measure total FLOPs for the full sequence (prompt + generation)
        total_inputs = tokenizer("a" * (prompt_len + gen_len), return_tensors="pt")
        total_flops, _, _ = calculate_flops(model=model, kwargs=total_inputs, print_results=False, output_as_string=False)
        
        # The total decode FLOPs is the total FLOPs minus the prefill FLOPs.
        total_decode_flops = total_flops - prefill_flops
    else:
        total_decode_flops = 0.0

    return prefill_flops, total_decode_flops

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
        gt_prefill, gt_total_decode = get_ground_truth_flops(model, tokenizer, prompt_len, gen_len)
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
    model_name = "google/gemma-2b"

    # Define the number of points to use for calibration in each test run.
    # Each tuple means (num_points_for_prefill_fit, num_points_for_decode_fit)
    calibration_configs = [
        (3, 2),  # Use 3 points to fit prefill, 2 to fit decode
        (6, 4),
        (12, 8),
        (24, 16),
        (48, 32),
        (96, 64),
        (192, 128),
        (384, 256),
        (768, 512),
    ]

    # This is now our master dataset of possible prompt/generation lengths.
    # This dataset is verified to be safe for models with a context size of 8192 tokens.
    # For every pair (P, G), the condition P + G <= 8192 is met.
    full_dataset = [
        # --- 1. Tiny Prompts & Responses (Edge Cases & Startup Cost) ---
        (1, 1), (2, 2), (3, 5), (5, 3), (8, 1), (1, 8),
        (5, 10), (10, 5), (12, 12), (15, 20), (20, 15),

        # --- 2. Common Chat & Q/A ---
        # (Short-to-medium prompt, short-to-medium response)
        (30, 80),      # A quick question and a direct answer
        (45, 100),     # A follow-up question
        (60, 120),     # A more detailed question and a paragraph response
        (75, 40),      # Asking for a specific, short piece of information
        (90, 150),     # Multi-turn conversation history with a new question
        (100, 80),     # A concise question getting a direct answer
        (128, 128),    # Balanced Q/A
        (150, 100),    # A detailed prompt asking for a concise summary
        (200, 250),    # Providing context and getting a detailed answer
        (250, 50),     # A paragraph prompt asking for a single entity or fact
        (300, 200),    # A moderate conversation thread

        # --- 3. Instruction Following & Few-Shot Learning ---
        # (Medium prompt with examples, medium response)
        (256, 256),    # A balanced instruction-following task
        (350, 150),    # Detailed instructions for a specific format output
        (400, 400),    # Providing a few examples (few-shot) and a new query
        (512, 200),    # A standard "role-play" or persona prompt
        (600, 300),    # Complex instructions with constraints
        (750, 250),    # A prompt with a couple of text examples to learn from

        # --- 4. Code Generation & Explanation ---
        # (Highly variable ratios)
        (40, 200),     # "Write a Python function for X" -> function with docstrings
        (80, 400),     # "Create a simple HTML page with this structure"
        (250, 800),    # "Here's my class, add these methods and explain them"
        (500, 100),    # "Explain what this short script does"
        (800, 150),    # "Explain what this complex function does"
        (1024, 512),   # Refactoring a moderately-sized function
        (1200, 1000),  # "Refactor this large piece of legacy code"
        (1500, 200),   # "Find the bug in this code snippet"

        # --- 5. Summarization & Analysis ---
        # (Long prompt, medium-to-short response)
        (1000, 200),   # Summarizing a blog post
        (1500, 250),   # Summarizing a few pages of a book
        (2048, 300),   # Analyzing a document the size of the old context limit
        (3000, 400),   # Summarizing a short research paper
        (4096, 500),   # Analyzing a more substantial document
        (5000, 350),   # Pulling key insights from a long article
        (6000, 300),   # Creating an executive summary from a long report

        # --- 6. Retrieval-Augmented Generation (RAG) ---
        # (Very long prompt with context, concise response)
        (2000, 100),   # Answering a question based on a few retrieved documents
        (3500, 150),   # Synthesizing an answer from several sources
        (4000, 80),    # Answering a question based on a large chunk of a knowledge base
        (5500, 200),   # Multi-hop question answering
        (6000, 100),   # Finding a specific fact within a massive context blob
        (7500, 120),   # Synthesizing an answer from many different sources
        (8000, 100),   # Pushing the context limit to get a very specific answer

        # --- 7. Long-Form Generation (Creative Writing, Content Creation) ---
        # (Short prompt, very long response)
        (10, 1000),    # "Write a short story about..."
        (25, 2048),    # "Write a blog post on the topic of..."
        (50, 3000),    # "Continue this story..."
        (100, 4000),   # A detailed outline for a chapter, asking the model to write it

        # --- 8. Boundary & Stress Tests (All sums are <= 8192) ---
        # (Testing the extremes and common context boundaries)
        (1, 1023),     # Total 1024
        (1023, 1),     # Total 1024
        (512, 512),    # Total 1024
        (1, 2047),     # Total 2048
        (2047, 1),     # Total 2048
        (1024, 1024),  # Total 2048
        (1, 4095),     # Total 4096
        (4095, 1),     # Total 4096
        (2048, 2048),  # Total 4096
        (3072, 1024),  # Total 4096
        (1024, 3072),  # Total 4096
        (1, 8191),     # Total 8192 (Max generation)
        (8191, 1),     # Total 8192 (Max prompt)
        (4096, 4096),  # Total 8192 (50/50 split)
        (6144, 2048),  # Total 8192 (75/25 split)
        (2048, 6144),  # Total 8192 (25/75 split)
        (8000, 192),   # Total 8192 (Near max prompt)
        (7800, 300),   # Total 8100 (High RAG)

        # --- 9. Grid-like Coverage (Filling in the gaps) ---
        # (A variety of points to ensure no large areas are untested)
        (180, 600),
        (450, 1200),
        (700, 700),
        (900, 350),
        (1100, 800),
        (1400, 1400),
        (1800, 600),
        (2200, 1800),
        (2800, 1000),
        (3200, 3200),
        (3800, 2000),
        (4500, 3000),
        (5500, 2500),
        (7000, 1100),
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