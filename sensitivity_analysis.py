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
import bitsandbytes.functional as F
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

def sensitivity_study(model, tokenizer, optimizer, saved_state_dict, gradients):
    clear_memory()

    # Sensitivity analysis on a larger set of weights

    percent_of_weights = 5
    zo_eps = 1e-3
    custom_load_state_dict(model, saved_state_dict)
    layer_sensitivity = []

    attack_args = {'idx' : [0], 'attack_bit' : 7}

    for name, param in model.named_parameters():
        if 'weight' in name and (('attn' in name) or ('mlp' in name)):
            if gradients[name] is not None and param.dtype in [torch.bfloat16, torch.int8, torch.uint8]:
                clear_memory()
                sensitivity = [name]
                orig_data = copy.deepcopy(model.state_dict()[name].data.detach())
                k_top =  int((percent_of_weights/100)*gradients[name].detach().view(-1).size()[0])
                
                if model.state_dict()[name].dtype in [torch.int8, torch.bfloat16]:
                    w = model.state_dict()[name].data.detach().reshape(-1)
                else:
                    w = F.dequantize_4bit(model.state_dict()[name], param.quant_state, quant_type=param.quant_type, blocksize=param.blocksize).to(torch.bfloat16).float().detach().view(-1)
                g = gradients[name].float().detach().reshape(-1)
                imp_score = importance_score(w, g, alpha=0.5)
            
                wval, w_idx = w.detach().abs().topk(k_top)  # topk weights by magnitude
                gval, g_idx = gradients[name].detach().abs().reshape(-1).topk(k_top)    # topk weights by gradients
                ival,i_idx  =  imp_score.topk(k_top)    # topk weights by importance score

                clear_memory()
                state_dict_copy = copy.deepcopy(saved_state_dict)

                # Flip by weight magnitudes
                if w.dtype == torch.int8:
                    w[w_idx] = flip_bits_in_tensor(w[w_idx], 7)
                else:
                    w[w_idx] = -1*w[w_idx]
                if param.dtype in [torch.bfloat16, torch.int8]:
                    state_dict_copy[name].data[:] = w.clone().detach().reshape(param.data.shape)[:] 
                else:
                    state_dict_copy[name].data[:] = F.quantize_4bit(w, quant_type=param.quant_type, blocksize=param.blocksize)[0]
                custom_load_state_dict_single_layer(model, state_dict_copy, name)

                l,p= mmlu_loss(model, tokenizer, optimizer, '',['astronomy'], mode='zo'), calculate_perplexity(model,tokenizer,wiki_data,size=2)
                sensitivity.append(l[0])
                sensitivity.append(p)

                print(name, "Magnitude based:",l[0],p)

                # Restore to original
                custom_load_state_dict_single_layer(model, saved_state_dict, name)

                clear_memory()

                # Flip by gradients
                if model.state_dict()[name].dtype in [torch.int8, torch.bfloat16]:
                    w = model.state_dict()[name].data.detach().reshape(-1)
                else:
                    w = F.dequantize_4bit(model.state_dict()[name], param.quant_state, quant_type=param.quant_type, blocksize=param.blocksize).to(torch.bfloat16).float().detach().view(-1)
                state_dict_copy = copy.deepcopy(saved_state_dict)
                if w.dtype == torch.int8:
                    w[g_idx] = flip_bits_in_tensor(w[g_idx], 7)
                else:
                    w[g_idx] = -1*w[g_idx]
                if param.dtype in [torch.bfloat16, torch.int8]:
                    state_dict_copy[name].data[:] = w.clone().detach().reshape(param.data.shape)[:] 
                else:
                    state_dict_copy[name].data[:] = F.quantize_4bit(w, quant_type=param.quant_type, blocksize=param.blocksize)[0]
                custom_load_state_dict_single_layer(model, state_dict_copy, name)

                l,p= mmlu_loss(model, tokenizer, optimizer, '',['astronomy'], mode='zo'), calculate_perplexity(model,tokenizer,wiki_data,size=2)
                sensitivity.append(l[0])
                sensitivity.append(p)

                print(name, "Gradient based:",l[0],p)

                # Restore to original state
                custom_load_state_dict_single_layer(model, saved_state_dict, name)
            
                clear_memory()

                # Flip bits by importance score
                if model.state_dict()[name].dtype in [torch.int8, torch.bfloat16]:
                    w = model.state_dict()[name].data.detach().reshape(-1)
                else:
                    w = F.dequantize_4bit(model.state_dict()[name], param.quant_state, quant_type=param.quant_type, blocksize=param.blocksize).to(torch.bfloat16).float().detach().view(-1)
                state_dict_copy = copy.deepcopy(saved_state_dict)
                if w.dtype == torch.int8:
                    w[i_idx] = flip_bits_in_tensor(w[i_idx], 7)
                else:
                    w[i_idx] = -1*w[i_idx]
                if param.dtype in [torch.bfloat16, torch.int8]:
                    state_dict_copy[name].data[:] = w.clone().detach().reshape(param.data.shape)[:] 
                else:
                    state_dict_copy[name].data[:] = F.quantize_4bit(w, quant_type=param.quant_type, blocksize=param.blocksize)[0]
                custom_load_state_dict_single_layer(model, state_dict_copy, name)

                l,p= mmlu_loss(model, tokenizer, optimizer, '',['astronomy'], mode='zo'), calculate_perplexity(model,tokenizer,wiki_data,size=2)
                sensitivity.append(l[0])
                sensitivity.append(p)
                print(name, "Gradient+magnitude based:",l[0],p)
            
                # Restore to original state
                custom_load_state_dict_single_layer(model, saved_state_dict, name) #revert changes
                clear_memory()
                layer_sensitivity.append(sensitivity)

    return layer_sensitivity

def sensitivity_ablation(model, tokenizer, optimizer, saved_state_dict, gradients):
    percent_of_weights = [0.00001, 0.0001, 0.001, 0.01, 0.1,1,5,10]
    zo_eps = 1e-3
    layer_sensitivity = {}
    clear_memory()
    custom_load_state_dict(model, saved_state_dict)
    #line 18
    for name, param in model.named_parameters():
        if (gradients[name] is not None) and ('weight' in name) and (('attn' in name) or ('mlp' in name)):# and 'language_model.model.embed_tokens' not in name:
            
            attack_args = {'idx' : [0], 'attack_bit' : 7}

            sensitivity = {'Magnitude':[], 'Gradient':[], 'Gradient+Magnitude':[], 'number_of_weights_flipped': [], 'percentage_of_weights_flipped': percent_of_weights}
    
            k_tops =  [int((k/100)*gradients[name].detach().view(-1).size()[0]) for k in percent_of_weights]
            print(name, k_tops)
            if model.state_dict()[name].dtype in [torch.bfloat16, torch.int8]:
                w = param.data.detach().contiguous().view(-1)
            else:
                w = F.dequantize_4bit(model.state_dict()[name], param.quant_state, quant_type=param.quant_type, blocksize=param.blocksize).to(torch.bfloat16).float().detach().view(-1)
            g = gradients[name].float().detach().view(-1)
            # print(w.shape)
    
            imp_score = importance_score(w, g, alpha=0.5)  

            print(f'Layer name: {name}')
    
            for k_top in k_tops:
                # BFLIP
                print(f'k_top: {k_top}')
                wval, w_idx = w.detach().abs().reshape(-1).topk(k_top)
                gval, g_idx = gradients[name].detach().abs().view(-1).topk(k_top)
                ival, i_idx  = imp_score.topk(k_top)
                clear_memory()
    
                state_dict_copy = copy.deepcopy(saved_state_dict)
                if param.dtype==torch.int8:
                    w[w_idx] = flip_bits_in_tensor(w[w_idx], 7)
                    # print(attack_args['idx'])
                    state_dict_copy[name].data[:] = w.reshape(state_dict_copy[name].data.shape)
                else:
                    # print(w[w_idx])
                    w[w_idx] = -w[w_idx]
                    if param.dtype == torch.uint8:
                        state_dict_copy[name].data[:] = F.quantize_4bit(w, quant_type=param.quant_type, blocksize=param.blocksize)[0].reshape(state_dict_copy[name].data.shape)
                    else:
                        state_dict_copy[name] = w.reshape(state_dict_copy[name].data.shape)
                custom_load_state_dict(model, state_dict_copy)
                clear_memory()
    
                l  = mmlu_loss(model, tokenizer, optimizer, '',['astronomy'],mode='zo')[0]
                sensitivity['Magnitude'].append(l)
    
                print(name, "Magnitude based:",l)
    
                custom_load_state_dict(model, saved_state_dict)
                clear_memory()
            
                if model.state_dict()[name].dtype in [torch.int8, torch.bfloat16]:
                    w = model.state_dict()[name].data.detach().reshape(-1)
                else:
                    w = F.dequantize_4bit(model.state_dict()[name], param.quant_state, quant_type=param.quant_type, blocksize=param.blocksize).to(torch.bfloat16).float().detach().view(-1)
    
                state_dict_copy = copy.deepcopy(saved_state_dict)
                if param.dtype==torch.int8:
                    w[g_idx] = flip_bits_in_tensor(w[g_idx], 7)
                    # print(attack_args['idx'])
                    state_dict_copy[name].data[:] = w.reshape(state_dict_copy[name].data.shape)
                else:
                    # print(w[w_idx])
                    w[g_idx] = -w[g_idx]
                    if param.dtype == torch.uint8:
                        state_dict_copy[name].data[:] = F.quantize_4bit(w, quant_type=param.quant_type, blocksize=param.blocksize)[0].reshape(state_dict_copy[name].data.shape)
                    else:
                        state_dict_copy[name] = w.reshape(state_dict_copy[name].data.shape)
                custom_load_state_dict(model, state_dict_copy)
                clear_memory()
    
                l= mmlu_loss(model, tokenizer, optimizer, '',['astronomy'],mode='zo')[0]
    
                sensitivity['Gradient'].append(l)
    
                print(name, "Gradient based:",l)
    
                custom_load_state_dict(model, saved_state_dict)
                clear_memory()
            
                if model.state_dict()[name].dtype in [torch.int8, torch.bfloat16]:
                    w = model.state_dict()[name].data.detach().reshape(-1)
                else:
                    w = F.dequantize_4bit(model.state_dict()[name], param.quant_state, quant_type=param.quant_type, blocksize=param.blocksize).to(torch.bfloat16).float().detach().view(-1)
    
                state_dict_copy = copy.deepcopy(saved_state_dict)
    
                state_dict_copy = copy.deepcopy(saved_state_dict)
                if param.dtype==torch.int8:
                    w[i_idx] = flip_bits_in_tensor(w[i_idx], 7)
                    # print(attack_args['idx'])
                    state_dict_copy[name].data[:] = w.reshape(state_dict_copy[name].data.shape)
                else:
                    # print(w[w_idx])
                    w[i_idx] = -w[i_idx]
                    if param.dtype == torch.uint8:
                        state_dict_copy[name].data[:] = F.quantize_4bit(w, quant_type=param.quant_type, blocksize=param.blocksize)[0].reshape(state_dict_copy[name].data.shape)
                    else:
                        state_dict_copy[name] = w.reshape(state_dict_copy[name].data.shape)
                custom_load_state_dict(model, state_dict_copy)
                clear_memory()
    
                l = mmlu_loss(model, tokenizer, optimizer, '',['astronomy'],mode='zo')[0]
            
                sensitivity['Gradient+Magnitude'].append(l)
            
                print(name, "Gradient+magnitude based:",l)
                custom_load_state_dict(model, saved_state_dict)
                sensitivity['number_of_weights_flipped'].append(k_top)
                clear_memory()
            layer_sensitivity[name] = sensitivity
    return layer_sensitivity

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
    print(gradients)

    # Clear cache and get original state dictionary
    clear_memory()
    saved_state_dict = copy.deepcopy(model.state_dict())

    # Clear Cache
    clear_memory()

    # Reload quantized model
    model, tokenizer = load_model(model_name, args.q)

    # Clear cache
    clear_memory()

    # Sensitivity analysis
    print(f'Sensitivity Analysis.......')
    layer_sensitivity = sensitivity_study(model, tokenizer, optimizer, saved_state_dict, gradients)
    clear_memory()

    # Sanity Check on model integrity
    print(f'Sanity check after sensitivity Analysis.......')
    custom_load_state_dict(model, saved_state_dict)
    perplexity, loss = calculate_perplexity(model,tokenizer,wiki_data, size=20), mmlu_loss(model, tokenizer, optimizer, '', ['astronomy'])[0]
    print(f'WikiText perplexity: {perplexity}')
    print(f'MMLU Loss: {loss}')

    # Plot layer sensitivity

    fig = plt.figure(1)
    
    x =range(len(layer_sensitivity))
    
    y1, y2, y3 = [],[], []
    for i in layer_sensitivity:
        y1.append(i[1])
        y2.append(i[3])
        y3.append(i[5])
    
    plt.scatter(x, y1, c ="blue", label='Magnitude based attack')
    plt.scatter(x, y2, c ="red", label='Gradient based attack')
    # plt.scatter(x, y3, c ="green", label='Magnitude+Gradient based attack')
    plt.xlabel("Layer Id")
    plt.ylabel("Model loss")
    # To show the plot
    plt.legend()
    plt.grid()
    plt.show()

    df = pd.DataFrame()
    df['Layer name'] = []
    df['Magnitude attack loss'] = []
    df['Magnitude attack perplexity'] = []
    df['gradient attack loss'] = []
    df['gradient attack perplexity'] = []
    df['magnitude_+_Gradient attack loss'] = []
    df['magnitude_+_Gradient attack perplexity'] = []
    for i in layer_sensitivity:
        df.loc[len(df.index)] = [i[0], i[1], i[2], i[3], i[4],i[5],i[6]]

    print(df)

    df.to_csv(f'./log_results/{model_id}_fp16grad_sensitivity_5%_sign_bit_flip_v2.csv')
    k=20
    df2 = df.sort_values(by=['Magnitude attack loss'], ascending=False)[:k]
    df1 = df.sort_values(by=['gradient attack loss'], ascending=False)[:k]
    df3 = df.sort_values(by=['magnitude_+_Gradient attack loss'], ascending=False)[:k]
    df4 = pd.merge(df1, df2, how ='inner')
    top=df3[df3.columns[0]][:5].tolist()
    print(top)

    print(f'Sensitivity Ablation.......')
    layer_sensitivity = sensitivity_ablation(model, tokenizer, optimizer, saved_state_dict, gradients)
    clear_memory()

    # Store ablations
    flattened_data = []
    for layer, sensitivities in layer_sensitivity.items():
        percent_of_weights_flipped = sensitivities['percentage_of_weights_flipped']
        number_of_weights_flipped = sensitivities['number_of_weights_flipped']
        for sensitivity_type, values in sensitivities.items():
            if sensitivity_type not in ['percentage_of_weights_flipped', 'number_of_weights_flipped']:
                for index, value in enumerate(values):
                    flattened_data.append({
                        'Layer': layer,
                        'Attack_type': sensitivity_type,
                        'percentage_of_weights_flipped': percent_of_weights_flipped[index],
                        'number_of_weights_flipped': number_of_weights_flipped[index],
                        'Model_loss': value
                    })
    # Create a DataFrame
    df = pd.DataFrame(flattened_data)
    # Save to CSV
    df.to_csv(f'./log_results/{model_id}_layers_sensitivity_full_sign_bit_flip.csv', index=False)
    print(df)

    df = pd.read_csv(f'./log_results/{model_id}_layers_sensitivity_full_sign_bit_flip.csv')
    # df = df[df['percentage_of_weights_flipped'] == 5]
    df1 = df[df['Attack_type']=='Gradient']
    df2 = df[df['Attack_type']=='Magnitude']
    df3 = df[df['Attack_type']=='Gradient+Magnitude']

    dtypes = []
    for i in range(df3.shape[0]):
        if model.state_dict()[df3['Layer'].iloc[i]].dtype == torch.int8:
            dtypes.append('int8')
        else:
            dtypes.append('int8')

    df3['dtype'] = dtypes
    df3_n = df3[df3['dtype'] == 'int8']

    plt.figure(figsize=(8, 6))
    marker = itertools.cycle((',', '+', '.', 'o', '*', 'x', 'h','^')) 
    plt.rcParams.update({'font.size':14})

    # plt.scatter(list(range(1, df1.shape[0]+1)), df1['Model_loss'], color='b', label='Absolute Gradient', alpha=0.5)
    # plt.scatter(list(range(1, df2.shape[0]+1)), df2['Model_loss'], color='g', label='Absolute Magnitude', alpha=0.5)
    plt.scatter(list(range(1, df3_n.shape[0]+1)), df3_n['Model_loss'], color='r', label='Hybrid Sensitivity Score')
    plt.ylim([1.3, 10])
    plt.grid()
    plt.legend()
    plt.xlabel('Layer ID')
    plt.ylabel('Model Loss')
    plt.savefig(f'./log_results/{model_id}_layer_sens_5.pdf')
    plt.show()

    int_8_layers = []
    for name, param in model.named_parameters():
        if param.dtype==torch.int8 and gradients[name] is not None:
            int_8_layers.append(name)

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

    # Sanity Check on model integrity
    print(f'Sanity check after sensitivity ablation.......')
    custom_load_state_dict(model, saved_state_dict)
    perplexity, loss = calculate_perplexity(model,tokenizer,wiki_data, size=20), mmlu_loss(model, tokenizer, optimizer, '', ['astronomy'])[0]
    print(f'WikiText perplexity: {perplexity}')
    print(f'MMLU Loss: {loss}')

    del model, saved_state_dict, tokenizer, gradients

    print('Hello World')