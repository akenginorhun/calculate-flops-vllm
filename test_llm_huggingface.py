# !usr/bin/env python
# -*- coding:utf-8 -*-

'''
 Description  : 
 Version      : 1.0
 Author       : MrYXJ
 Mail         : yxj2017@gmail.com
 Github       : https://github.com/MrYxJ
 Date         : 2023-09-03 11:21:30
 LastEditTime : 2023-09-09 00:56:46
 Copyright (C) 2023 mryxj. All rights reserved.
'''
import os
import argparse
from calflops import calculate_flops_hf

def main():
    parser = argparse.ArgumentParser(description='Calculate FLOPs for Hugging Face models')
    parser.add_argument('--model', type=str, default="Qwen/Qwen3-8B", 
                        help='Model name from Hugging Face Hub (default: Qwen/Qwen3-8B)')
    args = parser.parse_args()

    batch_size = 1
    max_seq_length = 128
    # model_name = "baichuan-inc/Baichuan-13B-Chat"
    # flops, macs, params = calculate_flops_hf(model_name=model_name,
    #                                          input_shape=(batch_size, max_seq_length))
    # print("%s FLOPs:%s  MACs:%s  Params:%s \n" %(model_name, flops, macs, params))

    access_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    model_name = args.model
    flops, macs, params, print_results = calculate_flops_hf(model_name=model_name,
                                                            input_shape=(batch_size, max_seq_length),
                                                            forward_mode="forward",
                                                            print_results=False,
                                                            return_results=True,
                                                            access_token=access_token)

    print(print_results)
    print("%s FLOPs:%s  MACs:%s  Params:%s \n" %(model_name, flops, macs, params))

if __name__ == "__main__":
    main()
