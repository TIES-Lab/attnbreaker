# 🧠 AttentionBreaker: A Framework for Parameter Set Optimization for Bit-Flip Attacks in Transformers

This repository provides a comprehensive framework for fault injection, sensitivity analysis, and optimization targeting transformer-based language models. It includes tools for Bit-Flip Attacks (BFA), layer-wise sensitivity scoring, and ablation studies to assess model robustness and performance degradation.

## 📁 Repository Structure

```
.
├── ablation_study.py            # Script to evaluate accuracy under various ablation strategies
├── attn_breaker_utils.py        # Utility functions for sensitivity metrics and tensor manipulation
├── genbfa_optimization.py       # Optimization and search strategy for generalized bit-flip attack
├── genbfa_utils.py              # Helper functions for BFA (e.g., bit-level manipulation)
├── sensitivity_analysis.py      # Main script to rank layers/parameters by sensitivity to faults
├── requirements.txt             # List of requirements for the framework operations
```

## 🚀 Key Features

- **Layer-wise Sensitivity Scoring** using gradient and weight norms
- **Generalized Bit-Flip Attack (BFA)** with search-based optimization
- **Modular Bit Manipulation Utilities**
- **Ablation & Recovery Experiments** to test model robustness
- **Integration-ready pipeline** with HuggingFace models and quantized weights

## 🛠️ Setup

```bash
git clone https://github.com/TIES-Lab/attnbreaker.git
cd attentionbreaker
pip install -r requirements.txt
```

## 📌 Usage

### 1. Sensitivity Analysis

```bash
python sensitivity_analysis.py -m meta-llama/Llama-3.1-8B-Instruct --q int8 --d cuda:0
```

Ranks layers by their vulnerability to bit-flips based on combined gradient and weight magnitudes for a huggingface model at a given quantization level. Quantizations supported at INT8 and NF4 by BitsAndBytes. Default device is the CPU.

### 2. Ablation Study

```bash
python ablation_study.py -m meta-llama/Llama-3.1-8B-Instruct --q int8 --d cuda:0
```

Ablates top-k sensitive layers and evaluates BFAs for different sinsitivity scores and layer selections.

### 3. Generalized BFA Optimization

```bash
python genbfa_optimization.py -m meta-llama/Llama-3.1-8B-Instruct --q int8 --d cuda:0
```

Performs optimization after sensitivity analysis and ablation studies to reduce the critical parameter set for a BFA using evolutionary optimization.

## 📄 Citation

If you use this codebase in your research or publication, please consider citing:

```bibtex
@article{das2024attentionbreaker,
  title={GenBFA: An Evolutionary Optimization Approach to Bit-Flip Attacks on LLMs},
  author={Das, Sanjay and Bhattacharya, Swastik and Kundu, Souvik and Kundu, Shamik and Menon, Anand and Raha, Arnab and Basu, Kanad},
  journal={arXiv preprint arXiv:2411.13757},
  year={2024}
}
```

## 🧩 Acknowledgements

Built using [HuggingFace Transformers](https://github.com/huggingface/transformers) and inspired by recent work in fault injection and robustness analysis of LLMs.
