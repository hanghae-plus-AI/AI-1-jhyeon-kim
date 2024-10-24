import os
import sys
import torch
import wandb
import logging
import datasets
import argparse
import transformers

from typing import Optional
from itertools import chain
from dataclasses import dataclass, field

from datasets import load_dataset
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
wandb.init(project='Hanghae99')
wandb.run.name = 'gpt-finetuning-with-validation'

@dataclass
class Arguments:
    model_name_or_path: Optional[str] = field(default=None)
    torch_dtype: Optional[str] = field(default=None, metadata={'choices': ['auto', 'bfloat16', 'float16', 'float32']})
    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    block_size: int = field(default=1024)  # Block size 추가
    num_workers: Optional[int] = field(default=None)
    validation_split_percentage: Optional[int] = field(default=10)

# Parse arguments
parser = HfArgumentParser((Arguments, TrainingArguments))
args, training_args = parser.parse_args_into_dataclasses()

# 명시적으로 TrainingArguments 설정
# 명시적으로 TrainingArguments 설정
training_args.do_train = True  # 학습 수행
training_args.do_eval = True  # 평가 수행
training_args.evaluation_strategy = "steps"  # 스텝마다 평가
training_args.eval_strategy = "steps"
training_args.eval_steps = 100  # 100 스텝마다 평가
training_args.logging_strategy = "steps"  # 스텝마다 로깅
training_args.logging_steps = 100  # 100 스텝마다 로깅
training_args.save_strategy = "steps"  # 스텝마다 모델 저장
training_args.save_steps = 100  # 100 스텝마다 저장
training_args.load_best_model_at_end = True  # 학습이 끝나면 가장 좋은 모델 로드
training_args.metric_for_best_model = "eval_loss"  # eval_loss로 가장 좋은 모델 결정
training_args.run_name = "gpt-finetuning-experiment-small-steps"  # 명시적으로 run_name 설정



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

# Load the dataset and split it into train and validation
raw_datasets = load_dataset(
    args.dataset_name,
    args.dataset_config_name
)

# Train-validation split
split_dataset = raw_datasets["train"].train_test_split(test_size=args.validation_split_percentage / 100)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

logger.info(f"Train dataset size: {len(train_dataset)}")
logger.info(f"Eval dataset size: {len(eval_dataset)}")

config = AutoConfig.from_pretrained(args.model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    config=config,
    torch_dtype=args.torch_dtype
)

tokenizer.pad_token_id = tokenizer.eos_token_id

embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))

column_names = list(raw_datasets["train"].features)
text_column_name = "text" if "text" in column_names else column_names[0]

def tokenize_function(examples):
    output = tokenizer(examples[text_column_name])
    return output

# Tokenize dataset
with training_args.main_process_first(desc="dataset map tokenization"):
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.num_workers,
        remove_columns=column_names
    )

# Block size 설정 (max_position_embeddings와 args.block_size 중 작은 값으로 설정)
block_size = min(args.block_size, config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 1024)

def group_texts(examples):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size

    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

# Group texts
with training_args.main_process_first(desc="grouping texts together"):
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.num_workers
    )

# Trainer 정의에 validation 데이터셋 추가
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["test"],  # Validation 데이터셋 추가
    tokenizer=tokenizer,
    data_collator=default_data_collator
)

checkpoint = None
last_checkpoint = get_last_checkpoint(training_args.output_dir)
if training_args.resume_from_checkpoint is not None:
    checkpoint = training_args.resume_from_checkpoint
else:
    checkpoint = last_checkpoint

# Train 모델
train_result = trainer.train(resume_from_checkpoint=checkpoint)

# Train 완료 후 모델 저장
trainer.save_model()

# Metrics 계산 및 wandb에 로그
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

# Validation 후 로그 기록 (step마다 자동으로 수행)
logger.info("Starting evaluation...")
eval_metrics = trainer.evaluate()  # Validation loss 평가
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
