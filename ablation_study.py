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
import attn_breaker_utils

import argparse

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
    attn_breaker_utils.model_repo_name = args.m
    
    notebook_login()

    clear_memory()

    model_name = args.m
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

    # dtypes = []
    # for i in range(df3.shape[0]):
    #     if model.state_dict()[df3['Layer'].iloc[i]].dtype == torch.int8:
    #         dtypes.append('int8')
    #     else:
    #         dtypes.append('bfloat16')

    # df3['dtype'] = dtypes
    # print(df3)
    # df3_n = df3[df3['dtype'] == 'int8']

    # int_8_layers = []
    int_8_layers = df3['Layer'].unique().tolist()
    # print(int_8_layers)
    # for name, param in model.named_parameters():
    #     if param.dtype==torch.bfloat16 and gradients[name] is not None:
    #         int_8_layers.append(name)

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

    # Population space with weights for most sensitive layer
    print(f'Getting population space with weights for most sensitive layer......')
    pop_space_top1 = gen_pop_space_top_layers(model, gradients, [sorted_top_5_layers['Layer'].iloc[0]], 2)

    # Population space with weights for most sensitive layer with alpha ablations
    print(f'Getting population space with weights for most sensitive layer with alpha ablations......')
    pop_space_top1 = gen_pop_space_top_layers_alpha(model, gradients, [sorted_top_5_layers['Layer'].iloc[0]], 2)

    # Get population spaces
    print(f'Getting population spaces......')
    sol_space_any = gen_pop_space_top_layers(model, gradients, list(sorted_layers['Layer']), 2) # all layers
    print(f'Weights from all layers: {len(sol_space_any)}')
    sol_space_1 = gen_pop_space_top_layers(model, gradients, list(sorted_top_5_layers['Layer'].iloc[0:1]), 2) # most sensitive
    print(f'Weights from most sensitive layer: {len(sol_space_1)}')
    sol_space_3 = gen_pop_space_top_layers(model, gradients, list(sorted_top_5_layers['Layer'].iloc[0:3]), 2) # top3
    print(f'Weights from top 3 most sensitive layer: {len(sol_space_3)}')
    sol_space_5 = gen_pop_space_top_layers(model, gradients, list(sorted_top_5_layers['Layer'].iloc[0:5]), 2) # top5
    print(f'Weights from top 5 most sensitive layer: {len(sol_space_5)}')

    # Normalization method on most sensitive layer
    print(f'Normalization method ablation on most sensitive layer........')
    clear_memory()

    loss_mag = []
    loss_grad = []
    loss_imp = []
    loss_sum_imp = []
    loss_grad_imp = []
    custom_load_state_dict(model, saved_state_dict)

    num_weights = logarithmic_indices(1000, len(sol_space_1))
    print(num_weights)

    for n in num_weights:
        print(f'num weights = {n}')
        clear_memory()
        sol_space_1.sort(key=lambda x: x[2])
        sol_space_1 = sol_space_1[::-1]
        loss_mag.append(attack_get_loss(model, saved_state_dict, tokenizer, optimizer, sol_space_1[:n], 1, 7)[0])

        clear_memory()
        sol_space_1.sort(key=lambda x: x[4])
        sol_space_1 = sol_space_1[::-1]
        loss_grad.append(attack_get_loss(model, saved_state_dict, tokenizer, optimizer, sol_space_1[:n], 3, 7)[0])

        clear_memory()
        sol_space_1.sort(key=lambda x: x[6])
        sol_space_1 = sol_space_1[::-1]
        loss_imp.append(attack_get_loss(model, saved_state_dict, tokenizer, optimizer, sol_space_1[:n], 5, 7)[0])

        clear_memory()
        sol_space_1.sort(key=lambda x: x[8])
        sol_space_1 = sol_space_1[::-1]
        loss_sum_imp.append(attack_get_loss(model, saved_state_dict, tokenizer, optimizer, sol_space_1[:n], 7, 7)[0])

        clear_memory()
        sol_space_1.sort(key=lambda x: x[10])
        sol_space_1 = sol_space_1[::-1]
        loss_grad_imp.append(attack_get_loss(model, saved_state_dict, tokenizer, optimizer, sol_space_1[:n], 9, 7)[0])

    loss_df = pd.DataFrame({'Num_Weights':num_weights, 'Magnitude': loss_mag, 'Gradient': loss_grad, 'Min_Max_Importance': loss_imp, 'Sum_Importance': loss_sum_imp, 'Grad_Importance':loss_grad_imp})
    loss_df.to_csv(f'./log_results/{model_id}_importance_var.csv', index=False)
    print(loss_df)

    marker = itertools.cycle(['.', '+', '*', '^'])
    plt.figure(figsize=(8,6))
    plt.rcParams.update({'font.size':14})

    plt.plot(loss_df['Num_Weights']*100/model.get_memory_footprint()/8, loss_df['Magnitude'], marker = next(marker), label = 'Absolute Magnitude')
    plt.plot(loss_df['Num_Weights']*100/model.get_memory_footprint()/8, loss_df['Gradient'], marker = next(marker), label = 'Absolute Gradient')
    plt.plot(loss_df['Num_Weights']*100/model.get_memory_footprint()/8, loss_df['Min_Max_Importance'], marker = next(marker), label = 'Min-Max Normalization')
    plt.plot(loss_df['Num_Weights']*100/model.get_memory_footprint()/8, loss_df['Grad_Importance'], marker = next(marker), label = 'Sum Normalization')
    plt.plot(loss_df['Num_Weights']*100/model.get_memory_footprint()/8, loss_df['Sum_Importance'], marker = next(marker), label = 'Gradient-Weighted \nMagnitude Score')
    plt.xlabel('Fault Rate (in %)')
    plt.ylabel('Model Loss')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./log_results/{model_id}_importance.pdf')
    plt.show()

    # Top layer sensitivity
    print(f'Check layer sensitivity........')
    loss_val_any = []
    loss_val_1 = []
    loss_val_3 = []
    loss_val_5 = []

    custom_load_state_dict(model, saved_state_dict)

    num_weights = [0]+logarithmic_indices(int(1e3), int(1e6))
    print(num_weights)
    index = 5

    for n in num_weights:
        print(f'num weights = {n}')

        sol_space_any.sort(key=lambda x: x[index+1])
        sol_space_any = sol_space_any[::-1]
        loss_val_any.append(attack_get_loss(model, saved_state_dict, tokenizer, optimizer, sol_space_any[:n], index)[0])

        sol_space_1.sort(key=lambda x: x[index+1])
        sol_space_1 = sol_space_1[::-1]
        loss_val_1.append(attack_get_loss(model, saved_state_dict, tokenizer, optimizer, sol_space_1[:n], index)[0])

        sol_space_3.sort(key=lambda x: x[index+1])
        sol_space_3 = sol_space_3[::-1]
        loss_val_3.append(attack_get_loss(model, saved_state_dict, tokenizer, optimizer, sol_space_3[:n], index)[0])

        sol_space_5.sort(key=lambda x: x[index+1])
        sol_space_5 = sol_space_5[::-1]
        loss_val_5.append(attack_get_loss(model, saved_state_dict, tokenizer, optimizer, sol_space_5[:n], index)[0])

    sol_space_loss_df = pd.DataFrame({'Num_Weights': num_weights, 'Sol_Space_Any': loss_val_any, 'Sol_Space_1': loss_val_1, 'Sol_Space_3': loss_val_3, 'Sol_Space_5': loss_val_5})
    sol_space_loss_df.to_csv(f'./log_results/{model_id}_sol_space_losses.csv')
    sol_space_loss_df = pd.read_csv(f'./log_results/{model_id}_sol_space_losses.csv').iloc[:, 1:]
    print(sol_space_loss_df)

    marker = itertools.cycle(['.', '+', '*', '^'])
    plt.figure(figsize=(8,6))
    plt.rcParams.update({'font.size':14})

    plt.plot(sol_space_loss_df['Num_Weights']*100/model.get_memory_footprint()/8, sol_space_loss_df['Sol_Space_Any'], marker = next(marker), label = 'All Layers')
    plt.plot(sol_space_loss_df['Num_Weights']*100/model.get_memory_footprint()/8, sol_space_loss_df['Sol_Space_1'], marker = next(marker), label = 'Top-1 Layer')
    plt.plot(sol_space_loss_df['Num_Weights']*100/model.get_memory_footprint()/8, sol_space_loss_df['Sol_Space_3'], marker = next(marker), label = 'Top-3 Layers')
    plt.plot(sol_space_loss_df['Num_Weights']*100/model.get_memory_footprint()/8, sol_space_loss_df['Sol_Space_5'], marker = next(marker), label = 'Top-5 Layers')
    plt.grid()
    plt.xlabel('Fault Rate (in %)')
    plt.ylabel('Model Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./log_results/{model_id}_topn_layers.pdf')
    plt.show()

    # alpha sweep
    print(f'Ablation study on the alpha values on the most sensitive layer using min-max normalization')
    loss_val_mag = []
    loss_val_grad = []
    loss_val_alpha0 = []
    loss_val_alpha025 = []
    loss_val_alpha05 = []
    loss_val_alpha075 = []
    loss_val_alpha1 = []

    pop_space = pop_space_top1

    for i in [0]+logarithmic_indices(int(1e2), len(pop_space))[:-1]:
        print(f'num weights = {i}')
        pop_space.sort(key=lambda x: x[2])
        pop_space = pop_space[::-1]
        loss_val_mag.append(attack_get_loss(model, saved_state_dict, tokenizer, optimizer, pop_space[:i], 1, attack_index=7)[0])

        pop_space.sort(key=lambda x: x[4])
        pop_space = pop_space[::-1]
        loss_val_grad.append(attack_get_loss(model, saved_state_dict, tokenizer, optimizer, pop_space[:i], 3, attack_index=7)[0])

        pop_space.sort(key=lambda x: x[6])
        pop_space = pop_space[::-1]
        loss_val_alpha0.append(attack_get_loss(model, saved_state_dict, tokenizer, optimizer, pop_space[:i], 5, attack_index=7)[0])

        pop_space.sort(key=lambda x: x[8])
        pop_space = pop_space[::-1]
        loss_val_alpha025.append(attack_get_loss(model, saved_state_dict, tokenizer, optimizer, pop_space[:i], 7, attack_index=7)[0])

        pop_space.sort(key=lambda x: x[10])
        pop_space = pop_space[::-1]
        loss_val_alpha05.append(attack_get_loss(model, saved_state_dict, tokenizer, optimizer, pop_space[:i], 9, attack_index=7)[0])

        pop_space.sort(key=lambda x: x[12])
        pop_space = pop_space[::-1]
        loss_val_alpha075.append(attack_get_loss(model, saved_state_dict, tokenizer, optimizer, pop_space[:i], 11, attack_index=7)[0])

        pop_space.sort(key=lambda x: x[14])
        pop_space = pop_space[::-1]
        loss_val_alpha1.append(attack_get_loss(model, saved_state_dict, tokenizer, optimizer, pop_space[:i], 13, attack_index=7)[0])

    mag_grad_loss_any = pd.DataFrame({'num_weights' : [0]+logarithmic_indices(int(1e2), len(pop_space))[:-1], 
                                'mag_loss' : loss_val_mag, 
                                'grad_loss' : loss_val_grad,
                                'alpha0_loss' : loss_val_alpha0,
                                'alpha025_loss' : loss_val_alpha025,
                                'alpha05_loss' : loss_val_alpha05,
                                'alpha075_loss' : loss_val_alpha075,
                                'alpha1_loss' : loss_val_alpha1,
                                })
    mag_grad_loss_any.to_csv(f'./log_results/{model_id}_fp16grad_popspacetop1_alpha_sweep.csv')

    marker = itertools.cycle((',', '+', '.', 'o', '*', 'x', 'h','^')) 
    pop_mag_df = pd.read_csv(f'./log_results/{model_id}_fp16grad_popspacetop1_alpha_sweep.csv')
    plt.figure(figsize=(8,6))
    plt.rcParams.update({'font.size':14})
    # plt.semilogx(pop_mag_df['num_weights'], pop_mag_df['mag_loss'], label = 'Magnitude-based attack', marker = next(marker))
    # plt.semilogx(pop_mag_df['num_weights'], pop_mag_df['grad_loss'], label = 'Gradient-based attack', marker = next(marker))
    plt.plot(pop_mag_df['num_weights']*100/model.get_memory_footprint()/8, pop_mag_df['alpha0_loss'], label = 'Alpha=0 attack', marker = next(marker))
    plt.plot(pop_mag_df['num_weights']*100/model.get_memory_footprint()/8, pop_mag_df['alpha025_loss'], label = 'Alpha=0.25 attack', marker = next(marker))
    plt.plot(pop_mag_df['num_weights']*100/model.get_memory_footprint()/8, pop_mag_df['alpha05_loss'], label = 'Alpha=0.5 attack', marker = next(marker))
    plt.plot(pop_mag_df['num_weights']*100/model.get_memory_footprint()/8, pop_mag_df['alpha075_loss'], label = 'Alpha=0.75 attack', marker = next(marker))
    plt.plot(pop_mag_df['num_weights']*100/model.get_memory_footprint()/8, pop_mag_df['alpha1_loss'], label = 'Alpha=1 attack', marker = next(marker))
    plt.xlabel('Fault Rate (in %)')
    plt.ylabel('Model Loss')
    plt.grid()
    plt.legend()
    plt.savefig(f'./log_results/{model_id}_alpha_sweep.pdf')
    plt.show()

    del model, saved_state_dict, tokenizer, gradients

    print('Hello World')