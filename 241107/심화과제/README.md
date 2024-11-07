## 학습 속도 개선 시도
### 시도: 한국어 감정분류 모델 파인튜닝 시 LoRA 적용한 버전과 미적용 버전으로 학습시켜보고 비교

### 결과 (base_model vs lora_model)
  - 학습 시간 : lora_model 이 더 빠름
    - base_model: 745.53
    - lora_model: 565.96
    
  - train/loss: base_model 이 더 낮음
    - https://wandb.ai/skjh0807s-hanghae99/korean_emotion_classification/reports/train-loss-24-11-08-03-30-15---VmlldzoxMDA3MzcyNQ?accessToken=t9ycypzj2efanbbwt9r3k9oi6hdievytws747hdqsqpvbq6n7ryxunu2oulazeo7 
    - base_model 이 더 낮은 loss (경량화 하지 않은 모델의 표현력이 더 좋은 것으로 보임)
    <img width="800" alt="image" src="https://github.com/user-attachments/assets/83f3e976-ae18-4bc9-bdc1-d9a9b3143aca">

  - 메모리 점유율:
    오히려 lora model 점유율이 더 높게 나타나는 현상..!
    <img width="1355" alt="image" src="https://github.com/user-attachments/assets/08786efd-f081-44b0-a85e-208fbb328b4f">

### 리뷰
  - lora_model 이 학습 시간은 더 작고, 표현력도 더 작을 것이라고는 예상했음.
  - 메모리 점유율에 대해서는 예상 못했던 현상. 중간에 lora rank 가 들어가면서 연산할 파라미터 수가 줄었을 것이라고 생각했는데 왜 이런 현상이 일어났는지 멘토링 때 질문하기
