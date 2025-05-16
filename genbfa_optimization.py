import warnings, torch
from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList,  AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login

import os
import torch
from datasets import load_dataset
import random
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    QuantoConfig, 
    AwqConfig,
    TrainingArguments,
    pipeline,
)
from bitsandbytes.optim import Adam8bit
import auto_gptq
import torch.nn as nn
import warnings

import numpy as np
import time, os, tqdm, json
import pandas as pd
import copy
import gc
from huggingface_hub import notebook_login

import matplotlib.pyplot as plt
import itertools

import math

from attn_breaker_utils import *
from genbfa_utils import *

import argparse
import genbfa_utils
import attn_breaker_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="For a huggingface model, at a given quantization level, perform attentionbreaker's sensitivity analysis"
    )
    # Positional argument: required
    parser.add_argument(
        "-m", 
        required=True,
        help="huggingface model repository"
    )
    parser.add_argument(
        "-q", 
        help="quantization level",
        default="int8"
    )
    # Optional flag that counts occurrences, e.g. -v, -vv
    parser.add_argument(
        "-d",
        default='cpu',
        help="Torch device"
    )

    args = parser.parse_args()

    warnings.filterwarnings('ignore')
    device = torch.device(args.d if torch.cuda.is_available() else "cpu")
    attn_breaker_utils.device = device
    genbfa_utils.device = device
    
    notebook_login()

    clear_memory()

    model_name = args.m
    attn_breaker_utils.model_repo_name = args.m
    model_id = args.m.translate(str.maketrans('/-.', '___')) + '_' + args.q
    model, tokenizer = load_model(model_name, args.q)

    print(f'Memory footprint of the model: {model.get_memory_footprint()}')

    wiki_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    perplexity = calculate_perplexity(model, tokenizer, wiki_data, size = 10)
    print(f'Perplexity: {perplexity}')

    # Optimizer
    optimizer = Adam8bit(model.parameters(), lr=2e-5)

    # Calculate MMLU Accuracy, loss and logits for test
    mmlu_test(model, tokenizer,'',['astronomy']) 
    loss, logits = mmlu_loss(model,tokenizer,optimizer,'',['astronomy'])
    print(f'Model loss: {loss}')

    gradients = get_gradients(model, args.q, tokenizer, optimizer)
    # print(gradients)

    # Clear cache and get original state dictionary
    clear_memory()
    saved_state_dict = copy.deepcopy(model.state_dict())

    # Clear Cache
    clear_memory()

    # Reload quantized model
    model, tokenizer = load_model(model_name, args.q)

    # Clear cache
    clear_memory()

    df = pd.read_csv(f'./log_results/{model_id}_layers_sensitivity_full_sign_bit_flip.csv')
    # df = df[df['percentage_of_weights_flipped'] == 5]
    df1 = df[df['Attack_type']=='Gradient']
    df2 = df[df['Attack_type']=='Magnitude']
    df3 = df[df['Attack_type']=='Gradient+Magnitude']

    int_8_layers = df3['Layer'].unique().tolist()

    # Get most sensitive layer
    df_int_8 = df3[df3['Layer'].isin(int_8_layers)]
    top_5_layers = df_int_8[df_int_8['Model_loss']>=4]
    sorted_top_5_layers = top_5_layers.sort_values(by=['Model_loss'],ascending=False)
    sorted_top_5_layers = sorted_top_5_layers.sort_values(by=['number_of_weights_flipped'],ascending=True)
    sorted_top_5_layers = sorted_top_5_layers.drop_duplicates(subset='Layer', keep='first')
    print(sorted_top_5_layers)

    df_int_8 = df[df['Layer'].isin(int_8_layers)]
    all_layers = df_int_8[df_int_8['Model_loss']>=4]
    sorted_layers = all_layers.sort_values(by=['Model_loss'],ascending=False)
    sorted_layers = sorted_layers.sort_values(by=['number_of_weights_flipped'],ascending=True)
    sorted_layers = sorted_layers.drop_duplicates(subset='Layer', keep='first')
    print(sorted_layers)
    sorted_layers['percentage_of_weights_flipped'] = sorted_layers['percentage_of_weights_flipped'].iloc[0]


    # Population space with weights for most sensitive layer with alpha ablations
    print(f'Getting population space with weights for most sensitive layer with alpha ablations......')
    pop_space_top1 = gen_pop_space_top_layers_alpha(model, gradients, [sorted_top_5_layers['Layer'].iloc[0]], 2)
    print(len(pop_space_top1))

    print(f'Performing evolutionary optimization......')
    optimized_solution, gen_df = evolutionary_optimization(model, model_id, tokenizer, optimizer, saved_state_dict, pop_space_top1)
    
    plt.rcParams.update({'font.size':14})
    fig, ax1 = plt.subplots(figsize=(8,6))

    ax1.plot(range(gen_df['Num_Generations'].shape[0]), gen_df['Num_Weights'], marker = '+', color='navy', label = 'Solution Length Progression')
    ax1.set_xlabel('Number of Generations')
    ax1.set_ylabel('Number of Weights')
    ax1.legend(loc = 'upper right', bbox_to_anchor=(1,0.9725))

    ax2 = ax1.twinx()
    ax2.plot(range(gen_df['Num_Generations'].shape[0]), gen_df['Loss'], marker = '+', color='tomato', label = 'Loss Progression')
    ax2.set_xlabel('Number of Generations')
    ax2.set_ylabel('Model Loss')
    ax2.legend(loc = 'upper right', bbox_to_anchor=(1,0.9))

    plt.grid()
    plt.tight_layout()
    plt.savefig(f'./log_results/{model_id}_genetic_sweep.pdf')
    plt.show()

    file_path = f'./log_results/{model_id}_critical_weights.json'
    with open(file_path, 'w') as file:
        json.dump(optimized_solution, file, indent=4)
    print(f'data saved in {file_path}')

    del model, saved_state_dict, tokenizer, gradients

    print('Hello World')