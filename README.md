# AGI 발현을 위한 메타인지 프레임워크 핵심기술 개발 및 실증
## AGI 발현을 위한 Verifier에 대한 연구 개발
### Meta Score의 불확실한 일부 지표에 의존적이지 않은 Ensemble Meta Scoring 기법
### 💡 예시
![image](./image/example.png)

## ⚙️ Requirements
To install requirements:
```
pip install -r requirements.txt
```

## 💻 Usage Guide
### 1. Ensemble Scorer 실행을 위한 Meta-Scorer 결과 파일 준비
Ensemble meta-score를 계산하기 위해서는 Meta-Scorer의 결과 파일이 필요합니다.
- [Meta-Scorer](https://github.com/HYU-AGI/Meta-Scorer) 를 참고해주세요. 결과 파일은 다음 경로 형식을 따라야 합니다: \
결과 파일은 다음 경로 형식을 따라야 합니다: \
```{meta_score_res_dir}/{dataset_name}/{meta_scoring}```

### 2. Ensemble Scorer 학습 (Training)
여러 meta-score를 종합적으로 학습하여 ensemble meta-score를 예측하는 모델을 학습합니다.
```
python src/ensemble_scoring.py --model_name "model_name" --dataset_name "dataset_name" --mode train --n_epochs 50 --testset_ratio 0.3
```
- ```model_name``` : 답변 생성에 사용했던 모델명
- ```dataset_name``` : 답변 생성에 사용했던 데이터셋명
- ```meta_score_res_dir``` : meta-score 결과 저장 디렉토리  
- ```ensemble_scorer_dir``` : Ensemble Scorer 학습 후 모델 파라미터 저장할 디렉토리
- ```testset_ratio``` : 학습에 사용되는 train/testset split을 위해 지정할 testset 비율

### 3. Ensemble Meta-score 계산 (Inference)

학습된 Ensemble Scorer를 이용해 최종 ensemble meta-score를 계산합니다.
```
python src/ensemble_scoring.py --model_name "model_name" --dataset_name "dataset_name" --mode ensemble_scoring
```
- ```model_name``` : 답변 생성에 사용했던 모델명
- ```dataset_name``` : 답변 생성에 사용했던 데이터셋명
- ```meta_score_res_dir``` : meta-score 결과 저장 디렉토리
- ```ensemble_scorer_dir``` : Ensemble Scorer 저장된 디렉토리

## 🧠 작동 원리
**1️⃣ 다양한 Meta-score를 하나의 Ensemble Score로 통합** \
여러 meta-score 중 일부는 불확실하거나 편향된 지표일 수 있습니다. \
이를 보완하기 위해, TabM 기반 모델을 활용하여 다양한 meta-score를 입력으로 받아 단일 ensemble meta-score로 통합합니다. \
이 모델은 학습 데이터를 통해 hallucination 탐지 능력을 향상시키도록 학습됩니다.

**2️⃣ Ensemble Meta-scoring 수행** \
학습된 모델 파라미터를 불러와 meta-score들을 종합 평가합니다. \
그 결과로 생성물의 hallucination 가능성을 나타내는 하나의 ensemble meta-score를 산출하게 됩니다.

**💡 장점**
- 다수의 meta-score를 통합하여 보다 신뢰도 높은 verification을 수행할 수 있습니다.
- 단일 지표에 의존하지 않는 평가 구조로, 편향된 metric의 영향을 최소화합니다.
- 학습 기반의 ensemble 구조를 통해 hallucination 검출 성능을 향상시킵니다.

### Reference
[TabM: Advancing tabular deep learning with parameter-efficient ensembling](https://openreview.net/pdf?id=Sd4wYYOhmY)
```
@inproceedings{gorishniytabm,
  title={TabM: Advancing tabular deep learning with parameter-efficient ensembling},
  author={Gorishniy, Yury and Kotelnikov, Akim and Babenko, Artem},
  booktitle={The Thirteenth International Conference on Learning Representations}
}
```
