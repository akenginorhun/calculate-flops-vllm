import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from calflops import calculate_flops

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
        tuple: A tuple containing the prefill and decode FLOPs.
    """
    # Prefill
    inputs = tokenizer("a" * prompt_len, return_tensors="pt")
    prefill_flops, _, _ = calculate_flops(model=model, kwargs=inputs, print_results=False)

    # Decode
    decode_flops_list = []
    for i in range(gen_len):
        inputs = tokenizer("a" * (prompt_len + i), return_tensors="pt")
        flops, _, _ = calculate_flops(model=model, kwargs=inputs, print_results=False)
        decode_flops_list.append(flops)
    
    decode_flops = np.mean(np.diff(decode_flops_list)) if len(decode_flops_list) > 1 else 0


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
        tuple: A tuple containing the fitted prefill and decode functions.
    """
    print("Running calibration...")
    # Prefill calibration
    prefill_x = np.array(prefill_points)
    prefill_y = np.array([get_ground_truth_flops(model, tokenizer, p, 1)[0] for p in prefill_points])
    prefill_coeffs = np.polyfit(prefill_x, prefill_y, 2)
    prefill_func = np.poly1d(prefill_coeffs)

    # Decode calibration
    decode_x = np.array(decode_points)
    decode_y = np.array([get_ground_truth_flops(model, tokenizer, 1, d)[1] for d in decode_points])
    decode_coeffs = np.polyfit(decode_x, decode_y, 1)
    decode_func = np.poly1d(decode_coeffs)
    
    print("Calibration complete.")
    return prefill_func, decode_func

def predict_flops(prefill_func, decode_func, prompt_len, gen_len):
    """
    Predicts FLOPs using the calibrated formulas.

    Args:
        prefill_func (np.poly1d): The fitted prefill function.
        decode_func (np.poly1d): The fitted decode function.
        prompt_len (int): The prompt length.
        gen_len (int): The generation length.

    Returns:
        tuple: A tuple containing the predicted prefill and decode FLOPs.
    """
    prefill_flops = prefill_func(prompt_len)
    decode_flops = decode_func(gen_len)
    return prefill_flops, decode_flops

def run_test(model, tokenizer, prefill_func, decode_func, test_grid):
    """
    Compares predictions from calibration against ground truth.

    Args:
        model (torch.nn.Module): The model to test.
        tokenizer: The tokenizer for the model.
        prefill_func (np.poly1d): The fitted prefill function.
        decode_func (np.poly1d): The fitted decode function.
        test_grid (list): A list of (prompt_len, gen_len) tuples for testing.
    """
    print("Running tests...")
    results = []
    for prompt_len, gen_len in test_grid:
        gt_prefill, gt_decode = get_ground_truth_flops(model, tokenizer, prompt_len, gen_len)
        pred_prefill, pred_decode = predict_flops(prefill_func, decode_func, prompt_len, gen_len)

        prefill_error = abs(pred_prefill - gt_prefill) / gt_prefill * 100
        decode_error = abs(pred_decode - gt_decode) / gt_decode * 100 if gt_decode > 0 else 0

        results.append({
            "prompt_len": prompt_len,
            "gen_len": gen_len,
            "gt_prefill": gt_prefill,
            "pred_prefill": pred_prefill,
            "prefill_error": prefill_error,
            "gt_decode": gt_decode,
            "pred_decode": pred_decode,
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

    print("\n--- Detailed Results ---")
    for r in results:
        print(
            f"Prompt: {r['prompt_len']}, Gen: {r['gen_len']} | "
            f"GT Prefill: {r['gt_prefill']:.2e}, Pred Prefill: {r['pred_prefill']:.2e}, Error: {r['prefill_error']:.2f}% | "
            f"GT Decode: {r['gt_decode']:.2e}, Pred Decode: {r['pred_decode']:.2e}, Error: {r['decode_error']:.2f}%"
        )

def main():
    """
    Main function to run the one-time calibration test.
    """
    # Parameters
    model_name = "sshleifer/tiny-gpt2"
    prefill_cal_points = [10, 50, 100]
    decode_cal_points = [10, 50, 100]
    test_grid = [(20, 30), (80, 60), (120, 90)]

    # Load model
    model, tokenizer = load_model(model_name)

    # Run calibration
    prefill_func, decode_func = calibrate(model, tokenizer, prefill_cal_points, decode_cal_points)

    # Run tests
    results = run_test(model, tokenizer, prefill_func, decode_func, test_grid)

    # Report results
    report_results(results)


if __name__ == "__main__":
    main()