# BASELINE 탐색

편향 보정을 적용하기 전, baseline 모델을 위한 하이퍼파라미터 튜닝을 진행한다. 

모델의 하이퍼파라미터는 `Validation` 데이터셋에 대해 `Recall@K`를 기반으로 평가한다. 

| param | list |
| :---: | :---: | 
| `k_dim` | 64, 128, 256 |
| `alpha` | 10.0, 20.0, 40.0 |
| `gamma` | 0.0 |
| `reg_weight` | 0.001, 0.01, 0.05 | 

앞서 말했듯이 편향 보정 이전의 baseline 모델을 대상으로 하기 때문에, 편향 보정 강도를 $\gamma = 0$ 로 고정하여 탐색을 진행한다. (`./Scripts/train.py grid_serach()`)

탐색 후,  다음과 같은 하이퍼파라미터 세팅에서 가장 높은 `Recall@10 = 0.01438155506005899`을 달성하였다. 

| param | value |
| :---: | :---: | 
| `k_dim` | 256 |
| `alpha` | 20.0 |
| `gamma` | 0.0 |
| `reg_weight` | 0.05 | 

따라서 위와 같은 경우에 학습된 모델을 이후 성능 비교에 사용할 baseline 으로 지정한다.  

해당 탐색 과정에서의 각 모델들의 성능 지표는 `./logs/grid_search_results.json` 에서 확인할 수 있다. 