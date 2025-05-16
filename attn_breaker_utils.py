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
import bitsandbytes.functional as F

import matplotlib.pyplot as plt
import itertools

import math

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model_repo_name = ""

TASKS = ['abstract_algebra', 
         'anatomy', 
         'astronomy', 
         'business_ethics', 
         'clinical_knowledge', 
         'college_biology', 
         'college_chemistry', 
         'college_computer_science', 
         'college_mathematics', 
         'college_medicine', 
         'college_physics', 
         'computer_security', 
         'conceptual_physics', 
         'econometrics', 
         'electrical_engineering', 
         'elementary_mathematics', 
         'formal_logic', 
         'global_facts', 
         'high_school_biology', 
         'high_school_chemistry', 
         'high_school_computer_science', 
         'high_school_european_history', 
         'high_school_geography', 
         'high_school_government_and_politics', 
         'high_school_macroeconomics', 
         'high_school_mathematics', 
         'high_school_microeconomics', 
         'high_school_physics', 
         'high_school_psychology', 
         'high_school_statistics', 
         'high_school_us_history', 
         'high_school_world_history', 
         'human_aging', 
         'human_sexuality', 
         'international_law', 
         'jurisprudence', 
         'logical_fallacies', 
         'machine_learning', 
         'management', 
         'marketing', 
         'medical_genetics', 
         'miscellaneous', 
         'moral_disputes', 
         'moral_scenarios', 
         'nutrition', 
         'philosophy', 
         'prehistory', 
         'professional_accounting', 
         'professional_law', 
         'professional_medicine', 
         'professional_psychology', 
         'public_relations', 
         'security_studies', 
         'sociology', 
         'us_foreign_policy', 
         'virology', 
         'world_religions'
         ]

# Choices in MMLU
choices = ["A", "B", "C", "D"]

def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()

def load_model(model_name, quantization):
    model_repo_name = model_name
    # Configure BitsAndBytes for int8 quantization
    if quantization == 'int8':
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,  # Set for 8-bit loading
            # bnb_8bit_quant_type=mode,  # Keep 'int8' mode (change to your preferred quantization method if any)
            # bnb_8bit_compute_dtype=torch.float16,  # Use float16 for computation, can change if needed
            # bnb_8bit_use_double_quant=True,  # Double quantization may improve performance
        )
        model = AutoModelForCausalLM.from_pretrained( 
        model_name,
        device_map=device,  # Automatically map model layers to devices
        quantization_config=bnb_config,
        # torch_dtype=torch.float16,  # float16 is standard for compute with quantization
        trust_remote_code=True
    )

    elif quantization == 'nf4':
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,               # enable 4-bit
            bnb_4bit_quant_type="nf4",       # specify NF4 quantization
            bnb_4bit_compute_dtype=torch.bfloat16,  # internal compute dtype
            bnb_4bit_use_double_quant=True   # usually boosts accuracy
        )
        model = AutoModelForCausalLM.from_pretrained( 
        model_name,
        device_map=device,  # Automatically map model layers to devices
        quantization_config=bnb_config,
        # torch_dtype=torch.float16,  # float16 is standard for compute with quantization
        trust_remote_code=True
    )

    else:
        bnb_config = BitsAndBytesConfig(
            # load_in_8bit=True,  # Set for 8-bit loading
            # bnb_8bit_quant_type=mode,  # Keep 'int8' mode (change to your preferred quantization method if any)
            # bnb_8bit_compute_dtype=torch.float16,  # Use float16 for computation, can change if needed
            # bnb_8bit_use_double_quant=True,  # Double quantization may improve performance
        )
    
        model = AutoModelForCausalLM.from_pretrained( 
            model_name,
            device_map=device,  # Automatically map model layers to devices
            # quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,  # float16 is standard for compute with quantization
            trust_remote_code=True
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
    tokenizer.pad_token = '[PAD]'
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer

def get_gradients(model, quantization, tokenizer, optimizer):
    gradients = {}

    if quantization == 'int8':
        # Disable quantization flag
        model.is_quantized = False

        # Cast quantized weights to float16
        for name, param in model.named_parameters():
            if param.dtype == torch.int8:
                param.data = param.data.to(torch.float16)
                param.requires_grad = True

        # Calculate loss with gradients
        mmlu_loss(model, tokenizer, optimizer, '',['astronomy'],mode='bp')

        # Store gradients
        for name, param in model.named_parameters():
            gradients[name] = param.grad
    
    elif quantization == 'nf4':
        # # Calculate loss with gradients

        model, tokenizer = load_model(model_repo_name, 'bfloat16')
        mmlu_loss(model, tokenizer, optimizer, '',['astronomy'],mode='bp')

        # Store gradients
        for name, param in model.named_parameters():
            gradients[name] = param.grad
        
        del model, tokenizer

    else:
        for name, param in model.named_parameters():
            print(name, param.dtype)
        #     if param.dtype == torch.bfloat16:
        #         param.requires_grad = True

        mmlu_loss(model, tokenizer, optimizer, '',['astronomy'],mode='bp')

        # Store gradients
        for name, param in model.named_parameters():
            gradients[name] = param.grad
    
    return gradients

def calculate_perplexity(model, tokenizer, dataset, size):
    total_loss = 0
    total_length = 0
    i = 0#random.randrange(0, len(dataset['text'])-size)
    for example in dataset['text'][i:size+i]:
        if example != '':
            input_text = example
            inputs = tokenizer(input_text, return_tensors="pt").to(device)
            # print(inputs)
            # print(input_text.dtype, '/',inputs)
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"].to(device))
            loss = outputs.loss
            loss = torch.nan_to_num(loss)
            total_loss += loss.item() * inputs["input_ids"].size(1)
            total_length += inputs["input_ids"].size(1)
            # print(example,'/', loss.item(), total_loss, total_length)
    
    perplexity = torch.exp(torch.tensor(total_loss / total_length))
    print(f"WikiText Perplexity: {perplexity:.4f}")
    return perplexity.item()

# Accuracy calculation
def compute_metric(output_filename):
    with open(output_filename, 'r') as f:
        run_results = json.load(f)
    total_acc = 0
    total_num = 0
    for task in run_results:
        acc = 0
        pred_answers = run_results[task]['pred_answers']
        gold_answers = run_results[task]['gold_answers']
        for pred, gold in zip(pred_answers, gold_answers):
            # print("pred: %s, gold: %s" % (pred, gold))
            if gold == pred.replace(' ', ''): acc += 1
        print("ACC-%s: %.4f" % (task, acc/len(gold_answers)))
        total_acc += acc
        total_num += len(gold_answers)
    print("ACC-all: %.4f" % (total_acc/total_num))
    return total_acc/total_num
 

# Format prompt subject 
def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

# Format prompts
def format_example(df, idx, subject, include_answer=True):
    prompt = "The following are multiple choice questions (with answers) about {}. ANSWER SHOULD BE IN ANY ONE OF A, B, C OR D. DO NOT ANSWER ANYTHING ELSE. THE ANSWER SHOULD ONLY BE A LETTER AND NOT A NUMBER\n".format(format_subject(subject))
    prompt += df.iloc[idx, 0]
    k = len(df['choices'].iloc[idx])
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df['choices'].iloc[idx][j])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(choices[df['answer'].iloc[idx]])
    return prompt
 

# Generate prompts from dataset
def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}. ANSWER SHOULD BE IN ANY ONE OF A, B, C OR D. DO NOT ANSWER ANYTHING ELSE. THE ANSWER SHOULD ONLY BE A LETTER AND NOT A NUMBER\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i, subject)
    return prompt
 
 
# Tokenize 
def prepare_input(tokenizer, prompts):
    tokenizer.pad_token = tokenizer.eos_token
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to('cuda')
 
    return input_tokens
 
# Split to different batches 
def batch_split(prompts, batch_num):
    batch_prompts = []
    mini_batch = []
    for prompt in prompts:
        mini_batch.append(prompt)
        if len(mini_batch) == batch_num:
            batch_prompts.append(mini_batch)
            mini_batch = []
    if len(mini_batch) != 0:
        batch_prompts.append(mini_batch)
    return batch_prompts

# Inference of split batches 
def batch_infer(model, tokenizer, prompts):
    batch_size = 1
    answers = []
    batch_prompts = batch_split(prompts, batch_size)
    for batch_input in batch_prompts:
        # print(batch_input)
        encode_inputs = prepare_input(tokenizer, batch_input).to(device)
        # print(f'{encode_inputs=}')
        outputs = model.generate(**encode_inputs, max_new_tokens=1, pad_token_id=128001, labels = encode_inputs['input_ids'])
        # print(f'{outputs=}')
        answers.append(tokenizer.batch_decode(outputs[0], skip_special_tokens=True)[-1])
        # print('answers: ', answers)
        # break
    # answers = [answer[-1] for answer in answers]
    return answers

# Inference for gradient calculation
def batch_infer_bp_loss(model, tokenizer, prompts, optimizer):
    batch_size = 4
    batch_prompts = batch_split(prompts, batch_size)
    for batch_input in batch_prompts:
        encode_inputs = prepare_input(tokenizer, batch_input).to(device)
        # print(encode_inputs)
        # st = time.time()
        outputs = model(**encode_inputs, labels=encode_inputs['input_ids'])
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        break

# Zeroth order loss calculation and logits
def batch_infer_zo_loss(model, tokenizer, prompts, optimizer):
    batch_size = 4
    accum_loss = 0
    i=0
    batch_prompts = batch_split(prompts, batch_size)
    for batch_input in batch_prompts:
        encode_inputs = prepare_input(tokenizer, batch_input).to(device)
        # print(encode_inputs)
        with torch.no_grad(): 
            outputs = model(**encode_inputs, labels=encode_inputs['input_ids'])
            # outputs = model(**encode_inputs)
        loss = outputs.loss
        i=i+1
        accum_loss = accum_loss+loss.item()
        break
    return loss.item(), outputs.logits

# Calculate MMLU Accuracy
def mmlu_test(model, tokenizer, file_name: str, TASKS):
   
    run_results = {}
    output_filename = 'run_results_%s.json' % (file_name)
    start_time = time.time()
    for task in TASKS:
        print('Testing %s ...' % task)
        records = []
        # dev_df = pd.read_csv(os.path.join("mmlu_data/", "dev", task + "_dev.csv"), header=None)[:5]    # Path to MMLU dataset as CSV
        # test_df = pd.read_csv(os.path.join("mmlu_data/", "test", task + "_test.csv"), header=None)     # Path to MMLU dataset as CSV
        
        splits = {'test': task+'/test-00000-of-00001.parquet', 'validation': task+'/validation-00000-of-00001.parquet', 'dev': task+'/dev-00000-of-00001.parquet'}
        dev_df = pd.read_parquet("hf://datasets/cais/mmlu/" + splits["dev"])[:5]
        test_df = pd.read_parquet("hf://datasets/cais/mmlu/" + splits["test"])
        # print(test_df)
        # break
        for i in range(test_df.shape[0]): # Change the number of iterations to limit the number of prompts for a particular task
            # get prompt and make sure it fits
            k = 5
            prompt_end = format_example(test_df, i, task, include_answer=False)
            # print(prompt_end)
            train_prompt = gen_prompt(dev_df, task, k)
            prompt = prompt_end
            while len(tokenizer.tokenize(prompt)) + 1> 2048: # bos token
                prompt_split = prompt.split("\n\n")
                prompt_split.pop(1)
                prompt = '\n\n'.join(prompt_split)
            print(prompt)     # Uncomment to print the prompts provided
            label = choices[test_df['answer'].iloc[i]]
            records.append({'prompt':prompt, 'answer':label})
 
        pred_answers = batch_infer(model, tokenizer, [record['prompt'] for record in records])
        
        gold_answers = [record['answer'] for record in records]
        run_results[task] = {'pred_answers':pred_answers, 'gold_answers':gold_answers}
        print(run_results)    # Uncomment to print run results
    with open(output_filename, 'w') as f:
        json.dump(run_results, f, ensure_ascii=False, indent=2)
   
    accuracy = compute_metric(output_filename)
    end_time = time.time()
    print("total run time %.2f" % (end_time - start_time))
    return accuracy

# Get MMLU loss
def mmlu_loss(model, tokenizer, optimizer, file_name: str, TASKS, mode = 'zo'):
   
    run_results = {}
    output_filename = 'run_results_%s.json' % (file_name)
    start_time = time.time()
    for task in TASKS:
        print('Testing %s ...' % task)
        records = []
        # dev_df = pd.read_csv(os.path.join("mmlu_data/", "dev", task + "_dev.csv"), header=None)[:5]     # Path to MMLU dataset as CSV
        # test_df = pd.read_csv(os.path.join("mmlu_data/", "test", task + "_test.csv"), header=None)      # Path to MMLU dataset as CSV
        splits = {'test': task+'/test-00000-of-00001.parquet', 'validation': task+'/validation-00000-of-00001.parquet', 'dev': task+'/dev-00000-of-00001.parquet'}
        dev_df = pd.read_parquet("hf://datasets/cais/mmlu/" + splits["dev"])[:5]
        test_df = pd.read_parquet("hf://datasets/cais/mmlu/" + splits["test"])
        # print(test_df)
        # break
        for i in range(test_df.shape[0]):
            # get prompt and make sure it fits
            k = 5
            prompt_end = format_example(test_df, i, task, include_answer=False)
            train_prompt = gen_prompt(dev_df, task, k)
            prompt = train_prompt + prompt_end
            while len(tokenizer.tokenize(prompt)) + 1> 2048: # bos token
                prompt_split = prompt.split("\n\n")
                prompt_split.pop(1)
                prompt = '\n\n'.join(prompt_split)
            label = choices[test_df['answer'].iloc[i]]
            records.append({'prompt':prompt, 'answer':label})
        
        if mode=='zo':
            loss, logits = batch_infer_zo_loss(model, tokenizer, [record['prompt'] for record in records], optimizer)
            return loss, logits
        elif mode =='bp':
            loss = batch_infer_bp_loss(model, tokenizer, [record['prompt'] for record in records], optimizer)
        
        return loss
    
#Function to flip bits
# BFLIP
def flip_bits_in_tensor(tensor, bit_position):
    bit_mask = 1 << bit_position
    flipped_tensor = tensor ^ bit_mask
    return flipped_tensor

# Functions to load state dictionary
def custom_load_state_dict(model, state_dict):
    model_state_dict = model.state_dict()
    for name, param in state_dict.items():
        if name in model_state_dict:
            # print(name)
            if 'weight_format' not in name:
                model_state_dict[name].copy_(param)
 
def custom_load_state_dict_single_layer(model, state_dict, layer_name):
    model_state_dict = model.state_dict()
    model_state_dict[layer_name].copy_(state_dict[layer_name].to(model_state_dict[layer_name].dtype))

# Function to calculate importance score using min-max normalization

#SSCORE
def importance_score(w, g, alpha=0.5):
    w_abs = w.detach().abs()
    w_min, w_max = w_abs.min(), w_abs.max()
    w_norm = (w_abs - w_min) / (w_max - w_min + 1e-8)  # Avoid division by zero

    # Normalize g (min-max normalization) in-place
    g_abs = g.detach().abs()
    g_min, g_max = g_abs.min(), g_abs.max()
    g_norm = (g_abs - g_min) / (g_max - g_min + 1e-8)  # Avoid division by zero

    # Compute score in a memory-efficient way using in-place operations
    score = (alpha * g_norm) + ((1 - alpha) * w_norm)

    return score

# Function to calculate importance score using gradient weighted normalization
def grad_weighted_mag_imp_score(w, g):
    # Normalize w (min-max normalization) in-place
    w_abs = w.detach().abs()
    w_min, w_max = w_abs.min(), w_abs.max()
    w_norm = (w_abs - w_min) / (w_max - w_min + 1e-8)  # Avoid division by zero

    # Normalize g (min-max normalization) in-place
    g_abs = g.detach().abs()
    g_min, g_max = g_abs.min(), g_abs.max()
    g_norm = (g_abs - g_min) / (g_max - g_min + 1e-8)  # Avoid division by zero

    # Compute score in a memory-efficient way using in-place operations
    score = g_norm * w_norm
    return score

# Function to calculate importance score using sum-normalization
def sum_norm_importance_score(w, g, alpha=0.5):
    w_abs = w.detach().abs()
    w_norm = w_abs/np.sum(w_abs.to(dtype=torch.float32).cpu().numpy())
    g_abs = g.detach().abs()
    g_norm = g_abs/np.sum(g_abs.to(dtype=torch.float32).cpu().numpy())  
    score = (alpha * g_norm) + ((1 - alpha) * w_norm)
 
    return score

def create_population_space(model, gradients, top_5_layers):
    solution_space = []
    for name, param in model.named_parameters():
        if param.dtype==torch.bfloat16 and gradients[name] is not None and name in top_5_layers['Layer'].unique():
            temp_df = top_5_layers[top_5_layers['Layer'] == name]
            k_top =  temp_df['number_of_weights_flipped'].item()
            print(name, k_top)
            if param.dtype==torch.bfloat16:
                w = param.data.detach().clone().view(-1)
            else:
                w = param.data.float().detach().view(-1)
            g = gradients[name].float().detach().view(-1)
 
            imp_score = importance_score(w, g, alpha=0.5)
 
            if temp_df['Attack_type'].item() == 'Magnitude':
                _, idx = w.detach().abs().topk(k_top)
            elif temp_df['Attack_type'].item() == 'Gradient':
                _, idx = gradients[name+'.weight'].detach().abs().view(-1).topk(k_top)
            elif temp_df['Attack_type'].item() == 'Gradient+Magnitude':
                _, idx  = imp_score.topk(k_top)
 
            for i in idx:
                solution_space.append([name, i.item()])
    return solution_space

def attack_get_loss_acc_new(curated_state_dict, model, gradients, tokenizer, optimizer, sol_set, attack_index=15, task = 'astronomy'):
    solutions = {}
    attack_args = {'idx' : [0], 'attack_bit' : attack_index}
    for sol in sol_set:
        key, value = sol[0],sol[1]
        if key not in solutions.keys():
            solutions[key] = [value]
        else:  
            solutions[key].append(value)
    # print(solutions)
    for name, param in model.named_parameters():
        if param.dtype==torch.bfloat16 and gradients[name] is not None and name in solutions.keys():
            # print(name, len(solutions[name]))
            if param.dtype==torch.bfloat16:
                w = param.data.detach().contiguous().view(-1)
            else:
                w = param.data.float().detach().view(-1)
            state_dict_copy = copy.deepcopy(curated_state_dict)
            idx = solutions[name]
            
            if param.dtype==torch.bfloat16:
                # print(attack_args['idx'])
                w[idx] = flip_bits_in_tensor(w[idx], attack_index)
                state_dict_copy[name].data[:] = w.reshape(param.data.shape)
            else:
                w[idx] = -w[idx]
                state_dict_copy[name+'.weight'] = w.reshape(param.data.shape)
            custom_load_state_dict(model, state_dict_copy)
            clear_memory()
 
    loss, acc = mmlu_loss(model, tokenizer, optimizer, '',[task],mode='zo')[0], mmlu_test(model, tokenizer, '',[task])
    custom_load_state_dict(model, curated_state_dict)
    return loss, acc*100

# Get loss with solution space indices only
def attack_get_loss_new(curated_state_dict, model, tokenizer, optimizer, sol_set, attack_index=7, task='astronomy'):
    solutions = {}
    attack_args = {'idx' : [0], 'attack_bit' : attack_index}
    for sol in sol_set:
        key, value = sol[0],sol[1]
        if key not in solutions.keys():
            solutions[key] = [value]
        else:  
            solutions[key].append(value)
    # print(solutions)
    for name, param in model.named_parameters():
        if param.dtype==torch.bfloat16 and name in solutions.keys():
            # print(name, len(solutions[name]))
            if param.dtype==torch.bfloat16:
                w = param.data.detach().contiguous().view(-1)
            else:
                w = param.data.float().detach().view(-1)
            state_dict_copy = copy.deepcopy(curated_state_dict)
            idx = solutions[name]
            
            if param.dtype==torch.bfloat16:
                # print(attack_args['idx'])
                w[idx] = flip_bits_in_tensor(w[idx], attack_index)
                state_dict_copy[name].data[:] = w.reshape(param.data.shape)
            else:
                w[idx] = -w[idx]
                state_dict_copy[name+'.weight'] = w.reshape(param.data.shape)
            custom_load_state_dict(model, state_dict_copy)
            clear_memory()
 
    loss= mmlu_loss(model, tokenizer, optimizer, '',[task],mode='zo')[0]
    custom_load_state_dict(model, curated_state_dict)
    return loss

# Generate population space with weight values across alpha ablations for min-max importance
def gen_pop_space(model,gradients, percent_of_weights = 0.01, grad_to_mag_ratio=1.0, type=1, random_flag = 0):
    # print("Attacked bit position: ", bit_position_to_flip, ", Percentage of weights attacked: ",percent_of_weights, ", Gradients to magnitude ratio:",grad_to_mag_ratio,',is random?:','yes' if random_flag else 'no'  )
    pop = []
    for name, param in model.named_parameters():
        if 'weight' in name and gradients[name] is not None and param.dtype==torch.bfloat16 and (('attn' in name) or ('mlp' in name)):
            clear_memory()
        # # custom_load_state_dict(model, curated_state_dict)
        # if 'weight' in name and gradients[name] is not None and param.dtype==torch.int32:
            print(name, param.dtype)
            # w1 = param.data
            # wf1 = torch.flatten(w1)
            # orig_dtype = wf1.dtype
 
            k_top =  int((percent_of_weights/100)*gradients[name].detach().abs().view(-1).size()[0])
            if param.dtype == torch.int8:
                w = param.data.detach().view(-1)
            elif param.dtype == torch.uint8:
                w = F.dequantize_4bit(model.state_dict()[name], param.quant_state, quant_type=param.quant_type, blocksize=param.blocksize).to(torch.bfloat16).float().detach().view(-1)
            else:
                w = param.data.detach().view(-1).to(dtype=torch.bfloat16)
            g = gradients[name].detach().abs().view(-1)
 
       
            wval, w_idx = w.topk(k_top)
            gval, g_idx = g.topk(k_top)
            alpha_0_val, alpha_0_idx = importance_score(w, g, alpha=0).topk(k_top)
            alpha_1_val, alpha_1_idx = importance_score(w, g, alpha=1).topk(k_top)
            alpha_025_val, alpha_025_idx = importance_score(w, g, alpha=0.25).topk(k_top)
            alpha_050_val, alpha_050_idx = importance_score(w, g, alpha=0.50).topk(k_top)
            alpha_075_val, alpha_075_idx = importance_score(w, g, alpha=0.75).topk(k_top)
            
            p = []
            for wid, w, gid, g, alpha0id, alpha0, alpha025id, alpha025, alpha050id, alpha050, alpha075id, alpha075, alpha1id, alpha1 in zip(w_idx, wval, g_idx, gval, alpha_0_idx, alpha_0_val, alpha_025_idx, alpha_025_val, alpha_050_idx, alpha_050_val, alpha_075_idx, alpha_075_val, alpha_1_idx, alpha_1_val):
                p.append([name, wid.item(),w.item(), gid.item(), g.item(), alpha0id.item(), alpha0.item(), alpha025id.item(), alpha025.item(), alpha050id.item(), alpha050.item(), alpha075id.item(), alpha075.item(), alpha1id.item(), alpha1.item()])
            pop=pop+p
            clear_memory()
            # custom_load_state_dict(model, saved_state_dict)
            # break
    return pop

# Generate population space with weight values only for some given layers
def gen_pop_space_top_layers(model,gradients, layers, percent_of_weights):
    # print("Attacked bit position: ", bit_position_to_flip, ", Percentage of weights attacked: ",percent_of_weights, ", Gradients to magnitude ratio:",grad_to_mag_ratio,',is random?:','yes' if random_flag else 'no'  )
    pop = []
    for name, param in model.named_parameters():
        if ('weight' in name) and (gradients[name] is not None) and (param.dtype in [torch.int8, torch.bfloat16, torch.uint8]) and (name in layers) and (('attn' in name) or ('mlp' in name)):
            clear_memory()
        # custom_load_state_dict(model, curated_state_dict)
        
 
            w1 = param.data
            wf1 = torch.flatten(w1)
 
            k_top =  int((percent_of_weights/100)*gradients[name].detach().abs().view(-1).size()[0])

            if param.dtype == torch.int8:
                w = param.data.detach().view(-1)
            elif param.dtype == torch.uint8:
                w = F.dequantize_4bit(model.state_dict()[name], param.quant_state, quant_type=param.quant_type, blocksize=param.blocksize).to(torch.bfloat16).float().detach().view(-1)
            else:
                w = param.data.detach().view(-1).to(dtype=torch.bfloat16)
            g = gradients[name].detach().abs().view(-1)
 
       
            wval, w_idx = w.detach().abs().topk(k_top)
            gval, g_idx = g.topk(k_top)
            i_val,i_idx = importance_score(w, g, alpha=0.5).topk(k_top)
            sum_imp_val, sum_imp_idx = sum_norm_importance_score(w, g).topk(k_top)
            grad_imp_val, grad_imp_idx = grad_weighted_mag_imp_score(w, g).topk(k_top)
            # alpha_025_val, alpha_025_idx = importance_score(w, g, alpha=0.25).topk(k_top)
            # alpha_050_val, alpha_050_idx = importance_score(w, g, alpha=0.50).topk(k_top)
            # alpha_075_val, alpha_075_idx = importance_score(w, g, alpha=0.75).topk(k_top)
            
            p = []
            # for wid, w, gid, g, alpha0id, alpha0, alpha025id, alpha025, alpha050id, alpha050, alpha075id, alpha075, alpha1id, alpha1 in zip(w_idx, wval, g_idx, gval, alpha_0_idx, alpha_0_val, alpha_025_idx, alpha_025_val, alpha_050_idx, alpha_050_val, alpha_075_idx, alpha_075_val, alpha_1_idx, alpha_1_val):
            #     p.append([name, wid.item(),w.item(), gid.item(), g.item(), alpha0id.item(), alpha0.item(), alpha025id.item(), alpha025.item(), alpha050id.item(), alpha050.item(), alpha075id.item(), alpha075.item(), alpha1id.item(), alpha1.item()])
            # pop=pop+p
            for wid, w, gid, g, iid, ival, sumid, sumval, gradid, gradval in zip(w_idx, wval, g_idx, gval, i_idx, i_val, sum_imp_idx, sum_imp_val, grad_imp_idx, grad_imp_val):
                p.append([name, wid.item(),w.item(), gid.item(), g.item(), iid.item(), ival.item(), sumid.item(), sumval.item(), gradid.item(), gradval.item()])
            pop=pop+p
            clear_memory()
            # custom_load_state_dict(model, curated_state_dict)
            # break
    return pop

# Generate population space with weight values only for some given layers with alpha ablations
def gen_pop_space_top_layers_alpha(model,gradients, layers, percent_of_weights):
    # print("Attacked bit position: ", bit_position_to_flip, ", Percentage of weights attacked: ",percent_of_weights, ", Gradients to magnitude ratio:",grad_to_mag_ratio,',is random?:','yes' if random_flag else 'no'  )
    pop = []
    for name, param in model.named_parameters():
        if 'weight' in name and gradients[name] is not None and (param.dtype in [torch.bfloat16, torch.int8, torch.uint8]) and name in layers and (('attn' in name) or ('mlp' in name)):
            clear_memory()
        # custom_load_state_dict(model, curated_state_dict)
        
 
            # w1 = param.data
            # wf1 = torch.flatten(w1)
 
            k_top =  int((percent_of_weights/100)*gradients[name].detach().abs().view(-1).size()[0])
            if param.dtype == torch.int8:
                w = param.data.detach().view(-1)
            elif param.dtype == torch.uint8:
                w = F.dequantize_4bit(model.state_dict()[name], param.quant_state, quant_type=param.quant_type, blocksize=param.blocksize).to(torch.bfloat16).float().detach().view(-1)
            else:
                w = param.data.detach().view(-1).to(dtype=torch.bfloat16)
            g = gradients[name].detach().abs().view(-1)
 
       
            wval, w_idx = w.detach().abs().topk(k_top)
            gval, g_idx = g.topk(k_top)
            # i_val,i_idx = importance_score(w, g, alpha=0.5).topk(k_top)
            # sum_imp_val, sum_imp_idx = sum_norm_importance_score(w, g).topk(k_top)
            # grad_imp_val, grad_imp_idx = grad_weighted_mag_imp_score(w, g).topk(k_top)
            alpha_025_val, alpha_025_idx = importance_score(w, g, alpha=0.25).topk(k_top)
            alpha_050_val, alpha_050_idx = importance_score(w, g, alpha=0.50).topk(k_top)
            alpha_075_val, alpha_075_idx = importance_score(w, g, alpha=0.75).topk(k_top)
            alpha_0_val, alpha_0_idx = importance_score(w, g, alpha=0).topk(k_top)
            alpha_1_val, alpha_1_idx = importance_score(w, g, alpha=1).topk(k_top)
            
            p = []
            for wid, w, gid, g, alpha0id, alpha0, alpha025id, alpha025, alpha050id, alpha050, alpha075id, alpha075, alpha1id, alpha1 in zip(w_idx, wval, g_idx, gval, alpha_0_idx, alpha_0_val, alpha_025_idx, alpha_025_val, alpha_050_idx, alpha_050_val, alpha_075_idx, alpha_075_val, alpha_1_idx, alpha_1_val):
                p.append([name, wid.item(),w.item(), gid.item(), g.item(), alpha0id.item(), alpha0.item(), alpha025id.item(), alpha025.item(), alpha050id.item(), alpha050.item(), alpha075id.item(), alpha075.item(), alpha1id.item(), alpha1.item()])
            pop=pop+p
            # for wid, w, gid, g, iid, ival, sumid, sumval, gradid, gradval in zip(w_idx, wval, g_idx, gval, i_idx, i_val, sum_imp_idx, sum_imp_val, grad_imp_idx, grad_imp_val):
            #     p.append([name, wid.item(),w.item(), gid.item(), g.item(), iid.item(), ival.item(), sumid.item(), sumval.item(), gradid.item(), gradval.item()])
            # pop=pop+p
            clear_memory()
            # custom_load_state_dict(model, curated_state_dict)
            # break
    return pop

# Get loss and logits after attack on given population space with weights, based on gradient, magnitude, or alpha ablation
def attack_get_loss(model, saved_state_dict, tokenizer, optimizer, indices, index, attack_index=15):
    solution = {}
    attack_args = {'idx' : [0], 'attack_bit' : attack_index}
    # model_state_dict = model.state_dict()
    state_dict_copy = copy.deepcopy(model.state_dict())
    for inner_list in indices:
        if isinstance(indices, list):
            key, value = inner_list[0], inner_list[index]
            if key not in solution:
                solution[key] = [value]
            else:
                solution[key].append(value)
    # print(state_dict_copy.keys())
    for name, param in model.named_parameters():
        if 'weight' in name and (param.dtype in [torch.bfloat16, torch.int8, torch.uint8]) and name in solution.keys():
            if param.dtype == torch.int8:
                w = param.data.detach().view(-1)
                w[solution[name]]= flip_bits_in_tensor(w[solution[name]], attack_index)
                state_dict_copy[name].data[:] = w.reshape(param.shape)
            elif param.dtype == torch.uint8:
                w = F.dequantize_4bit(model.state_dict()[name], param.quant_state, quant_type=param.quant_type, blocksize=param.blocksize).to(torch.bfloat16).float().detach().view(-1)
                w[solution[name]]= -1*w[solution[name]]
                state_dict_copy[name].data[:] = F.quantize_4bit(w, quant_type=param.quant_type, blocksize=param.blocksize)[0]
            else:
                w = param.data.detach().view(-1).to(dtype=torch.bfloat16)
                w[solution[name]]= -1*w[solution[name]]
            # w[solution[name]]= flip_bits_in_tensor(w[solution[name]], attack_index)
                state_dict_copy[name].data[:] = w.reshape(param.shape)
            custom_load_state_dict_single_layer(model, state_dict_copy, name)
    loss, logits = mmlu_loss(model, tokenizer, optimizer,'', ['astronomy'])
    print(loss)
    for name, param in model.named_parameters():
        if 'weight' in name and name in solution.keys():
            custom_load_state_dict_single_layer(model, saved_state_dict, name)

    for name, param in model.named_parameters():
        if 'weight' in name and name in solution.keys():
            custom_load_state_dict_single_layer(model, saved_state_dict, name)
    # model.load_state_dict(saved_state_dict)
    return loss, logits

# Get accuracy after attack on given population space with weights, based on gradient, magnitude, or alpha ablation
def attack_get_acc(model, saved_state_dict, tokenizer, indices, index, attack_index=15):
    solution = {}
    attack_args = {'idx' : [0], 'attack_bit' : attack_index}
    # model_state_dict = model.state_dict()
    state_dict_copy = copy.deepcopy(model.state_dict())
    for inner_list in indices:
        if isinstance(indices, list):
            key, value = inner_list[0], inner_list[index]
            if key not in solution:
                solution[key] = [value]
            else:
                solution[key].append(value)
    # print(state_dict_copy.keys())
    for name, param in model.named_parameters():
        if 'weight' in name and (param.dtype in [torch.bfloat16, torch.int8, torch.uint8]) and name in solution.keys():
            if param.dtype == torch.int8:
                w = param.data.detach().view(-1)
                w[solution[name]]= flip_bits_in_tensor(w[solution[name]], attack_index)
                state_dict_copy[name].data[:] = w.reshape(param.shape)
            elif param.dtype == torch.uint8:
                w = F.dequantize_4bit(model.state_dict()[name], param.quant_state, quant_type=param.quant_type, blocksize=param.blocksize).to(torch.bfloat16).float().detach().view(-1)
                w[solution[name]]= -1*w[solution[name]]
                state_dict_copy[name].data[:] = F.quantize_4bit(w, quant_type=param.quant_type, blocksize=param.blocksize)[0]
            else:
                w = param.data.detach().view(-1).to(dtype=torch.bfloat16)
                w[solution[name]]= -1*w[solution[name]]
            # w[solution[name]]= flip_bits_in_tensor(w[solution[name]], attack_index)
                state_dict_copy[name].data[:] = w.reshape(param.shape)
            custom_load_state_dict_single_layer(model, state_dict_copy, name)

    acc = mmlu_test(model, tokenizer,'', ['astronomy'])
    print(acc)
    for name, param in model.named_parameters():
        if 'weight' in name and name in solution.keys():
            custom_load_state_dict_single_layer(model, saved_state_dict, name)
    # model.load_state_dict(saved_state_dict)
    return acc

# Set of logarithmically spaced indices
def logarithmic_indices(start, x):
    # Ensure start is at least 0
    if start < 0:
        start = 0
    
    # Initialize indices list starting from the range [start, min(10, x)]
    indices = list(range(start, min(10, x+1)))
    
    power = 1
    while 10 ** power <= x:
        for i in range(1, 10):  # Add multiples of 10^power (10, 20, ..., 90)
            value = i * (10 ** power)
            if value > x:
                break
            if value >= start:
                indices.append(value)
        power += 1
    
    # Include the final value x if it's not already in the list
    if x >= start:
        indices.append(x)
        
    return sorted(set(indices))