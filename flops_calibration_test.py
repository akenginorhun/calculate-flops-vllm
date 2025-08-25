#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
One-time calibration test for estimating FLOPs of language models.

This script implements a one-time calibration approach:
1. Measures ground-truth FLOPs at various prompt/generation lengths using calflops
2. Fits simple formulas (quadratic for prefill, linear for decode) to calibration points
3. Validates predictions against ground truth on a test grid
4. Reports accuracy metrics
"""

import torch
import numpy as np
from scipy.optimize import curve_fit
from transformers import AutoTokenizer, AutoModelForCausalLM
from calflops import calculate_flops
import logging
import warnings
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class FLOPsCalibrator:
    """One-time calibration system for estimating FLOPs of language models."""
    
    def __init__(self, model_name: str, device: str = "cpu", trust_remote_code: bool = True):
        """
        Initialize the calibrator.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to run on ('cpu' or 'cuda')
            trust_remote_code: Whether to trust remote code for model loading
        """
        self.model_name = model_name
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.model = None
        self.tokenizer = None
        
        # Calibration data storage
        self.prefill_calibration_data = []
        self.decode_calibration_data = []
        
        # Fitted models
        self.prefill_model = None
        self.decode_model = None
        
    def load_model(self):
        """Load model and tokenizer."""
        logger.info(f"Loading model {self.model_name} on {self.device}...")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=torch.float32  # Use float32 for CPU
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code
        )
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded successfully")
        
    def measure_prefill_flops(self, prompt_lengths: List[int], batch_size: int = 1) -> List[Tuple[int, float]]:
        """
        Measure FLOPs for prefill at different prompt lengths.
        
        Args:
            prompt_lengths: List of sequence lengths to test
            batch_size: Batch size for testing
            
        Returns:
            List of (prompt_length, flops) tuples
        """
        logger.info(f"Measuring prefill FLOPs for prompt lengths: {prompt_lengths}")
        results = []
        
        for length in prompt_lengths:
            logger.info(f"Measuring prefill FLOPs for length {length}...")
            
            try:
                flops, _, _ = calculate_flops(
                    model=self.model,
                    input_shape=(batch_size, length),
                    transformer_tokenizer=self.tokenizer,
                    forward_mode="forward",
                    print_results=False,
                    output_as_string=False
                )
                results.append((length, flops))
                logger.info(f"Length {length}: {flops:.2e} FLOPs")
                
            except Exception as e:
                logger.error(f"Error measuring FLOPs for length {length}: {e}")
                continue
                
        return results
    
    def measure_decode_flops(self, generation_lengths: List[int], prompt_length: int = 32, batch_size: int = 1) -> List[Tuple[int, float]]:
        """
        Measure FLOPs for decode at different generation lengths.
        
        Args:
            generation_lengths: List of generation lengths to test
            prompt_length: Fixed prompt length for generation
            batch_size: Batch size for testing
            
        Returns:
            List of (generation_length, flops) tuples
        """
        logger.info(f"Measuring decode FLOPs for generation lengths: {generation_lengths}")
        results = []
        
        for gen_length in generation_lengths:
            logger.info(f"Measuring decode FLOPs for generation length {gen_length}...")
            
            try:
                # Create dummy input for generation
                input_ids = torch.ones((batch_size, prompt_length), dtype=torch.long, device=self.device)
                attention_mask = torch.ones((batch_size, prompt_length), dtype=torch.long, device=self.device)
                
                # Measure FLOPs using generate mode
                flops, _, _ = calculate_flops(
                    model=self.model,
                    args=[],
                    kwargs={
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                        'max_new_tokens': gen_length,
                        'do_sample': False,
                        'pad_token_id': self.tokenizer.pad_token_id
                    },
                    forward_mode="generate",
                    print_results=False,
                    output_as_string=False
                )
                
                results.append((gen_length, flops))
                logger.info(f"Generation length {gen_length}: {flops:.2e} FLOPs")
                
            except Exception as e:
                logger.error(f"Error measuring FLOPs for generation length {gen_length}: {e}")
                continue
                
        return results

    def fit_prefill_model(self, calibration_data: List[Tuple[int, float]]):
        """
        Fit quadratic model for prefill FLOPs.
        
        Args:
            calibration_data: List of (prompt_length, flops) tuples
        """
        if len(calibration_data) < 3:
            raise ValueError("Need at least 3 points to fit quadratic model")
            
        lengths = np.array([x[0] for x in calibration_data])
        flops = np.array([x[1] for x in calibration_data])
        
        # Quadratic function: f(x) = a*x^2 + b*x + c
        def quadratic(x, a, b, c):
            return a * x**2 + b * x + c
            
        try:
            popt, _ = curve_fit(quadratic, lengths, flops)
            self.prefill_model = lambda x: quadratic(x, *popt)
            self.prefill_coeffs = popt
            logger.info(f"Fitted prefill model: f(x) = {popt[0]:.2e}*x^2 + {popt[1]:.2e}*x + {popt[2]:.2e}")
        except Exception as e:
            logger.error(f"Failed to fit prefill model: {e}")
            raise
            
    def fit_decode_model(self, calibration_data: List[Tuple[int, float]]):
        """
        Fit linear model for decode FLOPs.
        
        Args:
            calibration_data: List of (generation_length, flops) tuples
        """
        if len(calibration_data) < 2:
            raise ValueError("Need at least 2 points to fit linear model")
            
        lengths = np.array([x[0] for x in calibration_data])
        flops = np.array([x[1] for x in calibration_data])
        
        # Linear function: f(x) = a*x + b
        def linear(x, a, b):
            return a * x + b
            
        try:
            popt, _ = curve_fit(linear, lengths, flops)
            self.decode_model = lambda x: linear(x, *popt)
            self.decode_coeffs = popt
            logger.info(f"Fitted decode model: f(x) = {popt[0]:.2e}*x + {popt[1]:.2e}")
        except Exception as e:
            logger.error(f"Failed to fit decode model: {e}")
            raise
    
    def predict_prefill_flops(self, prompt_length: int) -> float:
        """Predict prefill FLOPs for given prompt length."""
        if self.prefill_model is None:
            raise ValueError("Prefill model not fitted yet")
        return self.prefill_model(prompt_length)
    
    def predict_decode_flops(self, generation_length: int) -> float:
        """Predict decode FLOPs for given generation length."""
        if self.decode_model is None:
            raise ValueError("Decode model not fitted yet")
        return self.decode_model(generation_length)
    
    def run_calibration(self, 
                       prefill_lengths: List[int], 
                       decode_lengths: List[int],
                       decode_prompt_length: int = 32):
        """
        Run calibration process.
        
        Args:
            prefill_lengths: Prompt lengths for prefill calibration
            decode_lengths: Generation lengths for decode calibration  
            decode_prompt_length: Fixed prompt length for decode measurements
        """
        logger.info("Starting calibration process...")
        
        # Measure calibration data
        self.prefill_calibration_data = self.measure_prefill_flops(prefill_lengths)
        self.decode_calibration_data = self.measure_decode_flops(decode_lengths, decode_prompt_length)
        
        # Fit models
        if self.prefill_calibration_data:
            self.fit_prefill_model(self.prefill_calibration_data)
        if self.decode_calibration_data:
            self.fit_decode_model(self.decode_calibration_data)
            
        logger.info("Calibration completed")
    
    def evaluate_accuracy(self, 
                         test_prefill_lengths: List[int],
                         test_decode_lengths: List[int],
                         decode_prompt_length: int = 32) -> Dict[str, float]:
        """
        Evaluate calibration accuracy against ground truth.
        
        Args:
            test_prefill_lengths: Prompt lengths for testing prefill predictions
            test_decode_lengths: Generation lengths for testing decode predictions
            decode_prompt_length: Fixed prompt length for decode testing
            
        Returns:
            Dictionary with accuracy metrics
        """
        logger.info("Evaluating calibration accuracy...")
        
        results = {
            'prefill_avg_error': 0.0,
            'prefill_max_error': 0.0,
            'decode_avg_error': 0.0,
            'decode_max_error': 0.0
        }
        
        # Test prefill predictions
        if self.prefill_model and test_prefill_lengths:
            prefill_errors = []
            for length in test_prefill_lengths:
                try:
                    ground_truth = self.measure_prefill_flops([length])[0][1]
                    prediction = self.predict_prefill_flops(length)
                    error = abs(prediction - ground_truth) / ground_truth * 100
                    prefill_errors.append(error)
                    logger.info(f"Prefill length {length}: GT={ground_truth:.2e}, Pred={prediction:.2e}, Error={error:.2f}%")
                except:
                    logger.warning(f"Failed to test prefill length {length}")
                    
            if prefill_errors:
                results['prefill_avg_error'] = np.mean(prefill_errors)
                results['prefill_max_error'] = np.max(prefill_errors)
        
        # Test decode predictions
        if self.decode_model and test_decode_lengths:
            decode_errors = []
            for length in test_decode_lengths:
                try:
                    ground_truth = self.measure_decode_flops([length], decode_prompt_length)[0][1]
                    prediction = self.predict_decode_flops(length)
                    error = abs(prediction - ground_truth) / ground_truth * 100
                    decode_errors.append(error)
                    logger.info(f"Decode length {length}: GT={ground_truth:.2e}, Pred={prediction:.2e}, Error={error:.2f}%")
                except:
                    logger.warning(f"Failed to test decode length {length}")
                    
            if decode_errors:
                results['decode_avg_error'] = np.mean(decode_errors)
                results['decode_max_error'] = np.max(decode_errors)
        
        return results


def run_calibration_experiment(model_name: str,
                             prefill_calibration_points: List[int],
                             decode_calibration_points: List[int],
                             test_prefill_points: List[int],
                             test_decode_points: List[int],
                             device: str = "cpu") -> Dict[str, float]:
    """
    Run a complete calibration experiment.
    
    Args:
        model_name: HuggingFace model name
        prefill_calibration_points: Prompt lengths for prefill calibration
        decode_calibration_points: Generation lengths for decode calibration
        test_prefill_points: Prompt lengths for testing prefill predictions
        test_decode_points: Generation lengths for testing decode predictions
        device: Device to run on
        
    Returns:
        Dictionary with accuracy results
    """
    calibrator = FLOPsCalibrator(model_name, device)
    calibrator.load_model()
    
    # Run calibration
    calibrator.run_calibration(
        prefill_lengths=prefill_calibration_points,
        decode_lengths=decode_calibration_points
    )
    
    # Evaluate accuracy
    results = calibrator.evaluate_accuracy(
        test_prefill_lengths=test_prefill_points,
        test_decode_lengths=test_decode_points
    )
    
    return results


def main():
    """Main function to run calibration experiments with different configurations."""
    
    # Configuration parameters - easily adjustable
    MODEL_NAME = "microsoft/DialoGPT-small"  # Small model for quick testing
    DEVICE = "cpu"  # Force CPU for consistent results
    
    # Experiment configurations
    experiments = [
        {
            "name": "3 prefill + 2 decode",
            "prefill_cal": [16, 32, 64],
            "decode_cal": [8, 16],
            "prefill_test": [24, 48, 80],
            "decode_test": [12, 20]
        },
        {
            "name": "6 prefill + 4 decode", 
            "prefill_cal": [8, 16, 24, 32, 48, 64],
            "decode_cal": [4, 8, 12, 16],
            "prefill_test": [20, 40, 56],
            "decode_test": [6, 14, 18]
        }
    ]
    
    logger.info(f"Running calibration experiments with model: {MODEL_NAME}")
    logger.info(f"Device: {DEVICE}")
    
    # Run experiments
    for exp in experiments:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running experiment: {exp['name']}")
        logger.info(f"{'='*50}")
        
        try:
            start_time = time.time()
            results = run_calibration_experiment(
                model_name=MODEL_NAME,
                prefill_calibration_points=exp["prefill_cal"],
                decode_calibration_points=exp["decode_cal"],
                test_prefill_points=exp["prefill_test"],
                test_decode_points=exp["decode_test"],
                device=DEVICE
            )
            end_time = time.time()
            
            # Print results
            logger.info(f"\nResults for {exp['name']}:")
            logger.info(f"Prefill - Average Error: {results['prefill_avg_error']:.2f}%")
            logger.info(f"Prefill - Max Error: {results['prefill_max_error']:.2f}%")
            logger.info(f"Decode - Average Error: {results['decode_avg_error']:.2f}%")
            logger.info(f"Decode - Max Error: {results['decode_max_error']:.2f}%")
            logger.info(f"Experiment time: {end_time - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Experiment {exp['name']} failed: {e}")
            continue
    
    logger.info("\nAll experiments completed!")


if __name__ == "__main__":
    main()
