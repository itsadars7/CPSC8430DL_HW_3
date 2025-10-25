# Adarsha Neupane
# HW_3
# CPSC-8430
# Date: 10.23.2025

# BERT model fine-tuning on Spoken-SQuAD
# Implements: linear LR decay + warmup, doc_stride, fp16 AMP, gradient accumulation.

# Import necessary libraries
import os
import numpy as np
import collections
import random
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,                  # Automatically loads the correct tokenizer for pretrained model used
    AutoModelForQuestionAnswering,  # Automatically loads pretrained model for Question Answering tasks
    TrainingArguments,
    Trainer,
    default_data_collator,          # To handle batching for efficient GPU training 
    set_seed,
)
import evaluate

# Model Configuration
MODEL_NAME = "bert-base-uncased"        # BERT base model
# MODEL_NAME = "roberta-base"             # Roberta base model
# MODEL_NAME = "distilbert-base-uncased"  # Distilled BERT base model
# MODEL_NAME = "microsoft/deberta-base"   # Microsoft BERT base model
# MODEL_NAME = "microsoft/deberta-v3-base"   # Microsoft BERT V3 base model


TRAIN_FILE = "spoken_train-v1.1.json"
TEST_FILE   = "spoken_test-v1.1.json"  

MAX_LENGTH = 384                         # total sequence length (Q + context + specials)
DOC_STRIDE = 128                         # overlap between windows; < MAX_LENGTH
BATCH_SIZE = 8                           # per-device batch size
GRAD_ACCUM = 4                           # simulate global batch size = BATCH_SIZE * GRAD_ACCUM
LR         = 3e-5
NUM_EPOCHS = 2
WARMUP_RATIO = 0.1                       # 10% warmup then linear decay
WEIGHT_DECAY = 0.01
FP16 = True                              # AMP (needs a GPU that supports it)
SEED = 42
OUTPUT_DIR = "outputs_spoken_squad_1.1"
# OUTPUT_DIR = "outputs_spoken_squad_1.1_WER44"
# OUTPUT_DIR = "outputs_spoken_squad_1.1_WER54"


set_seed(SEED)

import json
from datasets import Dataset, DatasetDict

def read_squad(path):
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)

    rows = {"id": [], "question": [], "context": [], "answers": []}

    for article in j.get("data", []):
        for para in article.get("paragraphs", []):
            ctx = para.get("context", "")
            for qa in para.get("qas", []):        
                q_id = qa.get("id")
                q_text = (qa.get("question") or "").strip()
                ans_list = qa.get("answers", [])

                if ans_list:
                    # keep the first answer
                    a_text = ans_list[0].get("text", "")
                    a_start = ans_list[0].get("answer_start", 0)
                    answers = {"text": [a_text], "answer_start": [a_start]}
                else:
                    answers = {"text": [], "answer_start": []}

                rows["id"].append(q_id)
                rows["question"].append(q_text)
                rows["context"].append(ctx)
                rows["answers"].append(answers)

    return Dataset.from_dict(rows)


def load_spoken_squad(train_path, test_path):
    return DatasetDict({
        "train": read_squad(train_path),
        "test":  read_squad(test_path),
    })


def main():
    # Load training and test data
    data = load_spoken_squad(TRAIN_FILE, TEST_FILE)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True) # use_fast=True actually loads the high-speed tokenizer

    # Preprocess: training features
    def prepare_train_features(examples):
        # strip spaces from questions (helps tokenizer)
        questions = [q.strip() for q in examples["question"]]
        contexts = examples["context"]
        answers = examples["answers"]

        tokenized = tokenizer(
            questions,
            contexts,
            truncation="only_second",           # truncate context only
            max_length=MAX_LENGTH,
            stride=DOC_STRIDE,
            return_overflowing_tokens=True,     # split into multiple windows if the tokens > max_length
            return_offsets_mapping=True,        # for start and end character positions in the text
            padding="max_length",
        )

        sample_map = tokenized.pop("overflow_to_sample_mapping") # extracts mapping between example paragraph and multiple windows created from that example
        offsets = tokenized["offset_mapping"] # retrieves character start and end positions for each token

        start_positions = []
        end_positions = []

        for i, offset in enumerate(offsets):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]

            # If dataset has no answer, default to CLS
            if len(answer["answer_start"]) == 0:
                start_positions.append(0)
                end_positions.append(0)
                continue

            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])

            # Which tokens are context?
            seq_ids = tokenized.sequence_ids(i)
            
            # Find the start and end of the context in this feature
            ctx_start = 0
            while seq_ids[ctx_start] != 1:
                ctx_start += 1
            ctx_end = len(seq_ids) - 1
            while seq_ids[ctx_end] != 1:
                ctx_end -= 1
                
            # If answer not in this feature's span, set to CLS
            if not (offset[ctx_start][0] <= start_char and offset[ctx_end][1] >= end_char):
                start_positions.append(0)
                end_positions.append(0)
            else:
                # move start
                token_start = ctx_start
                while token_start <= ctx_end and offset[token_start][0] <= start_char:
                    token_start += 1
                # move end
                token_end = token_start - 1
                while token_end <= ctx_end and offset[token_end][1] < end_char:
                    token_end += 1
                start_positions.append(token_start - 1)
                end_positions.append(token_end)

        tokenized["start_positions"] = start_positions
        tokenized["end_positions"] = end_positions
        tokenized.pop("offset_mapping")  # not needed for training
        return tokenized

    # Preprocess: test features
    # Keep offsets for postprocessing
    def prepare_test_features(examples):
        questions = [q.strip() for q in examples["question"]]
        contexts = examples["context"]

        tokenized = tokenizer(
            questions,
            contexts,
            truncation="only_second",
            max_length=MAX_LENGTH,
            stride=DOC_STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = tokenized.pop("overflow_to_sample_mapping")
        tokenized["example_id"] = []

        for i in range(len(tokenized["input_ids"])):
            sample_idx = sample_map[i]
            tokenized["example_id"].append(examples["id"][sample_idx])

            # Only keep offsets for the context
            sequence_ids = tokenized.sequence_ids(i)
            tokenized["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None
                for k, o in enumerate(tokenized["offset_mapping"][i])
            ]
        return tokenized

    train_ds = data["train"].map(
        prepare_train_features,
        batched=True,
        remove_columns=data["train"].column_names,
    )
    test_ds = data["test"].map(
        prepare_test_features,
        batched=True,
        remove_columns=data["test"].column_names,
    )

    # Pretrained Model
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

    # Training args: linear LR decay + warmup, fp16, grad accumulation
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        save_total_limit=2,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="linear",
        warmup_ratio=WARMUP_RATIO,      
        fp16=FP16,
        logging_steps=100,
        report_to="none",
        seed=SEED,
    )

    # Evaluation Metrics (Exact Match and F1) 
    squad_metric = evaluate.load("squad")

    # Postprocess predictions into strings per example id
    def postprocess_predictions(preds, features, examples):
        start_logits, end_logits = preds       # logits predicted by BERT's QA output layer
        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}

        id_to_context = {ex_id: ctx for ex_id, ctx in zip(examples["id"], examples["context"])}
        
        features_per_example = collections.defaultdict(list)
        
        for i, ex_id in enumerate(features["example_id"]):
            features_per_example[ex_id].append(i)

        final_predictions = {}
        for ex_id, feat_idxs in features_per_example.items():
            best_text = ""
            best_score = -1e9

            for i in feat_idxs:
                offsets = features["offset_mapping"][i]
                s = int(np.argmax(start_logits[i]))
                e = int(np.argmax(end_logits[i]))

                # skip invalid spans
                if s > e or offsets[s] is None or offsets[e] is None:
                    continue

                start_char, end_char = offsets[s][0], offsets[e][1]
                # recover original context text
                # ex = data["test"][example_id_to_index[ex_id]]
                # context = ex["context"]
                context = id_to_context[ex_id]
                text = context[start_char:end_char]
                score = start_logits[i][s] + end_logits[i][e]
                if score > best_score:
                    best_score = score
                    best_text = text

            final_predictions[ex_id] = best_text
        return final_predictions

    def compute_metrics(eval_pred):
        preds, _ = eval_pred  # (start_logits, end_logits)
        pred_text = postprocess_predictions(preds, test_ds, data["test"])
        
        # Format ground-truth labels and model predictions (IDs and answers/predicted text) for metric computation
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in data["test"]]
        formatted_preds = [{"id": k, "prediction_text": v} for k, v in pred_text.items()]
        
        return squad_metric.compute(predictions=formatted_preds, references=references)

    
    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    # Train & Evaluate
    trainer.train()
    metrics = trainer.evaluate()
    print("\nTest metrics:", metrics)

    # Save final predictions to file
    
    print(">> starting predict")
    preds = trainer.predict(test_ds)
    
    print(">> predict done, starting postprocess")
    pred_text = postprocess_predictions(preds.predictions, test_ds, data["test"])
    print(">> postprocess done, writing file")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "predictions.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(pred_text, f, ensure_ascii=False, indent=2)
    print(f"Saved predictions to {os.path.join(OUTPUT_DIR, 'predictions.json')}")


if __name__ == "__main__":
    main()