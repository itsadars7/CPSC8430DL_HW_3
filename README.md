# CPSC8430DL_HW
# Adarsha Neupane


## Overview
This project fine-tunes BERT-based pretrained transformer-based Question Answering (QA) model:


The model is trained and evaluated on the **Spoken-SQuAD v1.1** dataset, which contains ASR-transcribed question-answering texts.  
Performance is evaluated using **Exact Match (EM)** and **F1 Score**.


## Dataset

| Split | Samples |
|-------|---------|
| Train | 37,111  |
| Test  | 5,351   |

Selected test set: `spoken_test-v1.1.json`  


## Training Setup

- **Hardware:** NVIDIA V100 GPU (Palmetto Cluster)
- **Epochs:** 2  
- **Batch size:** 8 (gradient accumulation 4 → global batch 32)
- **Learning rate:** 3e-5
- **Warmup:** 10% of total steps
- **Optimizer:** AdamW + linear LR decay
- **FP16:** Enabled  
- **Seed:** 42

### Tokenization Settings Tested
| max_length | doc_stride |
|-----------|------------|
| 384 | 128 |


## Results

| Model | max_length | doc_stride | EM | F1 |
|-------|------------|------------|----|----|
| BERT-base | 384 | 128 |  60.04 | 69.57 |

The final predictions are saved in:
outputs_spoken_squad_1.1/predictions.json
outputs_spoken_squad_1.1/metrics.json

## Running the Code

Ensure dependencies are installed:
```bash
pip install transformers datasets evaluate torch