import os
import sys
import torch
import wandb
import logging
import datasets
import argparse
import transformers
import json

from typing import Optional
from itertools import chain
from dataclasses import dataclass, field

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator
)
from transformers.trainer_utils import get_last_checkpoint

# Initialize wandb project
wandb.init(project='custom')
wandb.run.name = '1'

@dataclass
class Arguments:
    model_name_or_path: Optional[str] = field(default=None)
    torch_dtype: Optional[str] = field(default=None, metadata={'choices': ['auto', 'bfloat16', 'float16', 'float32']})
    # Removed dataset_name and dataset_config_name since we're using a local JSON file
    block_size: int = field(default=1024)  # Added block size
    num_workers: Optional[int] = field(default=None)
    validation_split_percentage: Optional[int] = field(default=10)
    json_file_path: Optional[str] = field(default=None, metadata={"help": "Path to the local JSON file"})

# Parse arguments
parser = HfArgumentParser((Arguments, TrainingArguments))
args, training_args = parser.parse_args_into_dataclasses()

# Explicitly set TrainingArguments
training_args.do_train = True
training_args.do_eval = True
training_args.evaluation_strategy = "steps"
training_args.eval_strategy = "steps"
training_args.eval_steps = 1
training_args.logging_strategy = "steps"
training_args.logging_steps = 1
training_args.save_strategy = "steps"
training_args.save_steps = 1
training_args.load_best_model_at_end = True
training_args.metric_for_best_model = "eval_loss"
training_args.run_name = "gpt-finetuning-experiment-small-steps"

# Logging setup
logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
if training_args.should_log:
    transformers.utils.logging.set_verbosity_info()

log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)

logger.info(f"Training/evaluation parameters {training_args}")

# Load the local JSON file
with open(args.json_file_path, 'r', encoding='utf-8') as f:
    json_data = json.load(f)

# Convert the JSON data into a list of dictionaries with 'text' key
texts = []
for item in json_data:
    instruction = item.get('instruction', '').strip()
    output = item.get('output', '').strip()
    # Combine instruction and output into a single text
    text = f"== {instruction} ==\n{output}\n"
    texts.append({'text': text})

# Create a Dataset from the list of texts
from datasets import Dataset

raw_dataset = Dataset.from_list(texts)

# Split the dataset into train and validation sets
split_dataset = raw_dataset.train_test_split(test_size=args.validation_split_percentage / 100)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

logger.info(f"Train dataset size: {len(train_dataset)}")
logger.info(f"Eval dataset size: {len(eval_dataset)}")

# Load model and tokenizer
config = AutoConfig.from_pretrained(args.model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    config=config,
    torch_dtype=args.torch_dtype
)

tokenizer.pad_token_id = tokenizer.eos_token_id

tokenizer.pad_token_id = tokenizer.eos_token_id

embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['text'])

# Tokenize datasets
with training_args.main_process_first(desc="dataset map tokenization"):
    tokenized_train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=args.num_workers,
        remove_columns=["text"]
    )
    tokenized_eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=args.num_workers,
        remove_columns=["text"]
    )


# Set block size
block_size = min(args.block_size, tokenizer.model_max_length, 128)  # Set a lower maximum block size


def group_texts(examples):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples['input_ids'])
    total_length = (total_length // block_size) * block_size

    result = {
        k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

# Group texts
with training_args.main_process_first(desc="grouping texts together"):
    lm_train_dataset = tokenized_train_dataset.map(
        group_texts,
        batched=True,
        num_proc=args.num_workers
    )
    lm_eval_dataset = tokenized_eval_dataset.map(
        group_texts,
        batched=True,
        num_proc=args.num_workers
    )

# Define Trainer with validation dataset
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_train_dataset,
    eval_dataset=lm_eval_dataset,
    tokenizer=tokenizer,
    data_collator=default_data_collator
)

checkpoint = None
last_checkpoint = get_last_checkpoint(training_args.output_dir)
if training_args.resume_from_checkpoint is not None:
    checkpoint = training_args.resume_from_checkpoint
elif last_checkpoint is not None:
    checkpoint = last_checkpoint

# Train the model
train_result = trainer.train(resume_from_checkpoint=checkpoint)

# Save the model after training
trainer.save_model()

# Compute metrics and log to wandb
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

# Evaluate and log results
logger.info("Starting evaluation...")
eval_metrics = trainer.evaluate()
logger.info(f"Evaluation metrics: {eval_metrics}")

if "eval_loss" in eval_metrics:
    wandb.log({
        "train/loss": metrics["train_loss"],
        "eval/loss": eval_metrics["eval_loss"]
    })
else:
    logger.warning("eval_loss not found in evaluation metrics!")
    wandb.log({
        "train/loss": metrics["train_loss"]
    })
