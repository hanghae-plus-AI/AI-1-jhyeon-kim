import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import torch
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import wandb

# argparse로 lora_r 값을 인자로 받음
parser = argparse.ArgumentParser(description="LoRA SFT Trainer with Custom LoRA Rank")
parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank value (e.g., 8, 128, 256)")
args = parser.parse_args()
lora_r = args.lora_r

# wandb 초기화 (각 LoRA Rank 값에 맞는 이름으로 초기화)
wandb.init(project="sft-lora-experiment", name=f"LoRA-tuning-rank-{lora_r}")

# 필요한 라이브러리들을 임포트합니다
# 데이터셋과 모델, 토크나이저를 로드합니다
dataset = load_dataset("lucasmccabe-lmi/CodeAlpaca-20k", split="train")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

# 데이터 포맷팅 함수 정의
def formatting_prompts_func(example):
    # 'instruction'과 'output'을 사용해 포맷팅된 텍스트 리스트를 만듭니다
    output_texts = [
        f"### Question: {instr}\n ### Answer: {out}"
        for instr, out in zip(example['instruction'], example['output'])
    ]
    return output_texts

# 응답 부분에 대한 구분자와 데이터 콜레이터 설정
response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# LoRA 설정 구성
lora_dropout = 0.1  # dropout 확률
lora_alpha = 32     # scaling factor
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=["q_proj", "v_proj"]  # 특정 모듈에만 LoRA 적용 (모델에 맞게 수정)
)

# LoRA를 모델에 적용
model_with_lora = get_peft_model(model, peft_config)

# SFTTrainer 구성
trainer = SFTTrainer(
    model=model_with_lora,
    train_dataset=dataset,
    args=SFTConfig(output_dir=f"/tmp/clm-instruction-tuning-r{lora_r}", max_seq_length=128),
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)

# 모델 학습 시작
trainer.train()

# wandb에 메트릭 로깅
wandb.log({
    "LoRA Rank": lora_r,
    "Max Memory Allocated (GB)": round(torch.cuda.max_memory_allocated(0)/1024**3, 1)
})

# 메모리 점유율 출력
print(f"LoRA Rank {lora_r} | Max Memory Allocated: {round(torch.cuda.max_memory_allocated(0)/1024**3, 1)} GB")

# wandb 종료
wandb.finish()
