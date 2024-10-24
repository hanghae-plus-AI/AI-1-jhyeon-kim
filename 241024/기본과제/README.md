# GPT 모델 파인튜닝 및 검증

이 프로젝트는 HuggingFace `transformers` 라이브러리를 사용하여 사전 학습된 GPT 모델을 파인튜닝합니다. 실험 추적을 위해 `wandb`를 활용하며, 학습 과정에서 일정한 간격으로 학습 손실(`train/loss`)과 검증 손실(`eval/loss`)을 기록합니다.

## 개요

이 스크립트는 지정된 데이터셋을 사용하여 GPT 모델을 파인튜닝합니다. 주요 기능은 다음과 같습니다:
- **스텝 기반 학습 및 평가**: 100 스텝마다 학습 손실과 검증 손실을 기록합니다.
- **최적의 모델 저장**: 학습이 완료되면 검증 성능이 가장 좋은 모델을 저장합니다.
- **`wandb`를 통한 실험 추적**: 실험 로그는 `wandb` 대시보드에서 실시간으로 확인 가능합니다.

## wandb 결과
- train/loss 링크: https://api.wandb.ai/links/skjh0807s-hanghae99/qri1yq1q
- eval/loss 링크: https://api.wandb.ai/links/skjh0807s-hanghae99/106yt9ox

### 이미지
<img width="1357" alt="image" src="https://github.com/user-attachments/assets/f4b05975-1320-47de-a0a0-2c6f7b7c2978">

<img width="1350" alt="image" src="https://github.com/user-attachments/assets/ffdb8822-66ec-46e3-a87c-76a021c41cd1">
