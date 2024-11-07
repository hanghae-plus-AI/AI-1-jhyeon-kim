## wandb 로그 - lora_r 따른 loss, 학습 속도, 그리고 메모리 점유율

### 1. loss
https://api.wandb.ai/links/skjh0807s-hanghae99/ztxwcc7e
<img width="900" alt="image" src="https://github.com/user-attachments/assets/26528ea4-39e6-4d8a-85f9-b3c020b4a354">


### 2. 학습속도(runtime) 및 메모리 점유율
<img width="376" alt="image" src="https://github.com/user-attachments/assets/3e389638-c057-47bd-99ec-db776835e8c9">
<img width="376" alt="image" src="https://github.com/user-attachments/assets/76ce27f0-fa3b-4183-9f74-2a87c4d5ce75">
<img width="364" alt="image" src="https://github.com/user-attachments/assets/a6254eb5-c085-475d-889a-9ff81dbe9ee4">

- 메모리 점유율: https://api.wandb.ai/links/skjh0807s-hanghae99/hx7my1pg
  <img width="900" alt="image" src="https://github.com/user-attachments/assets/0d09ca1a-dbe0-4e78-827a-4b005b48eb72">

### 3. 결과
#### 요약
```
- 메모리 점유율: rank 크기가 작을수록 메모리 사용량이 낮아짐.
- 학습 시간: rank 크기가 작을수록 학습 시간이 짧아짐.
- train/loss: rank 크기가 클수록 train/loss 값이 낮아짐.
```
#### 분석 및 인사이트
##### Rank와 모델 성능의 트레이드오프
```
rank를 낮추면 메모리 효율성과 학습속도는 증가하지만, 성능이 저하되는 트레이드오프가 발생할 수 있음. 
rank가 작으면 모델이 학습할 수 있는 정보의 범위가 제한되어 복잡한 패턴을 충분히 학습하기 어렵기 때문일 것.
```

##### 앞으로 고려사항
```
- 리소스 제한 환경: 주어진 환경에서 메모리와 학습 시간을 줄이기 위해 rank를 낮추는 것이 필요할 수 있음.
- 성능이 중요한 환경: 학습 시간이 길어지고 메모리 점유율이 높아지더라도, 높은 rank를 유지하는 것이 필요할 수 있음.
```
