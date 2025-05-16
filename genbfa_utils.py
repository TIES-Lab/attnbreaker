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

def union_mixed_type_lists(list1, list2):

    set1 = {tuple(item) for item in list1}
    set2 = {tuple(item) for item in list2}
    union_set = set1.union(set2)
    result = [list(item) for item in union_set]
    
    return result

def mutate(child, mutation_rate = 0.01):
    mutation_rate = random.randrange(1,10)/100
    temp = []
    for i in range(len(child)):
        if random.random() > mutation_rate:
            temp.append(child[i])
    return temp

def signcompare(l,l_th):
    if l >= l_th:
        return 1
    return -1

def crossover(parent1, parent2, SolutionSpace, crossover_prob=0.9):
    if random.random() < crossover_prob:
        child = []
        for gene1, gene2 in zip(parent1, parent2):
            if random.random() < 0.5:
                child.append(gene1)
            else:
                child.append(gene2)
    else:
        # child = list(random.sample(SolutionSpace, k=int(num_weights*0.99)))
        child = mutate(parent1)
    return child



def tournament_selection(sol_list):
    mating_pool = []
    while len(mating_pool) < 2:
        i, j = random.sample(range(len(sol_list)), 2)
        # print('Fitnesses:',sol_list[i][0], sol_list[j][0])
        if sol_list[i][0] > sol_list[j][0]:
            # print('returning:',sol_list[i][0])
            mating_pool.append(sol_list[i])
        else:
            # print('returning:',sol_list[j][0])
            mating_pool.append(sol_list[j])
    return mating_pool


def evolutionary_optimization(model, model_id, tokenizer, optimizer, saved_state_dict, pop_space_top1):
    InitNumSol = 50     #initial number of solutions
    max_gen = 150        #Maximum number of iterations/generations
    numSol = 40        #Number of solutions in each generation 
    index = 9
    pop_space_top1.sort(key=lambda x: x[index+1])
    pop_space_top1 = pop_space_top1[::-1]
    SolutionSpace = pop_space_top1[:]
    sol_list = [[0,SolutionSpace,0]]
    loss_progress = []
    loss_threshold  = 7
    clear_memory()

    for i in range(InitNumSol):
        sol_list.append([0, mutate(SolutionSpace),0])
    
    # print(sol_list)

    best_solution = sol_list[0]
    best_loss = 0
    l_progress = [] #For plotting
    opt_progress = []
    progress = {} #For plotting
    iterations_data = [] # snaping data from intermediate iterations
    interStep = 5 # iteration steps after snapping data

    custom_load_state_dict(model, saved_state_dict)
    while(max_gen>0):
        clear_memory()
        max_gen = max_gen - 1
        # Calculate loss per solution
        for i in range(len(sol_list)):
            # print(sol_list[i][1])
            loss = attack_get_loss(model, saved_state_dict, tokenizer, optimizer, sol_list[i][1], index)[0]
            # print(len(sol_list[i][1]))
            f = signcompare(loss, loss_threshold)*(loss/len(sol_list[i][1]))
            print(loss, len(sol_list[i][1]), f)
            sol_list[i][0] = f
            sol_list[i][2] = loss
            
        # rank the solutions based on their loss
        sol_list.sort(reverse = True)

            
        # collect progress information
        progress[-max_gen] = sol_list[0][2]

        if loss > best_loss:
            best_loss = loss
            best_solution = sol_list[0]
        best_solution = sol_list[0]
        # for i in range(len(sol_list)):
        #     if sol_list[i][0] > loss_threshold and len(sol_list[i][1]) < len(best_solution[1]):
        #         best_solution = sol_list[i]
        l_progress.append(len(best_solution[1]))
        opt_progress.append(best_solution[1])
        loss_progress.append(best_solution[2])
        loss = attack_get_loss(model, saved_state_dict, tokenizer, optimizer, best_solution[1], index)[0]
        print('-----------------Generation:',max_gen, 'Loss:', best_solution[0], 'Solution length:', len(best_solution[1]))


        sol_list2 = [best_solution,[0,mutate(best_solution[1], mutation_rate = 0.01), 0]] 


        for j in range(numSol-len(sol_list2)):
            #line 17 Algorithm 3 
            a1 = tournament_selection(sol_list)
            parent1 = a1[0][1]
            parent2 = a1[1][1]

            if random.random() > 0.5:
                child = crossover(best_solution[1], parent1, SolutionSpace, crossover_prob=0.9)
                child = [0,mutate(child, mutation_rate = 0.01), 0]
            else:
                child = crossover(best_solution[1], parent2, SolutionSpace, crossover_prob=0.9)
                
                child = [0,mutate(child, mutation_rate = 0.01), 0]
            
            sol_list2.append(child)

        # Use this solution list for next iteration
        sol_list = sol_list2

    attnbreaker_gen_df = pd.DataFrame({'Num_Generations':range(len(l_progress)), 'Num_Weights':l_progress, 'Loss':loss_progress})
    attnbreaker_gen_df.to_csv(f'./log_results/{model_id}.csv')
    return best_solution[1], attnbreaker_gen_df