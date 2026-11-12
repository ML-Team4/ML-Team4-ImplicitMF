# External Libraries or modules Details

실제 구현에서 사용된 외부 모듈, 클래스, 함수 등에 대한 설명을 진행한다. 

## `Implicit` 

> [implicit](https://benfred.github.io/implicit/)

Implicit-MF 를 WALS 를 통해 최적화를 진행하였고, 실제 훈련 단계에서 연산 속도 확보를 위해 `Implicit` 라이브러리를 사용하였다. 

### `implicit.als.AlternatingLeastSquares`

He et al. (2008) 의 Implicit ALS 알고리즘을 고속으로 수행하는, 행렬 인수분해 알고리즘 클래스이다. 

파라미터는 다음과 같다.  
- `factors` : 잠재 요인의 차원 수. default 100
- `regularization`: $L_2$ 정규화에 적용하는 가중치. default 0.01 
- `alpha` : 신뢰도 스케일링 가중치. default 1.0
- `iteration`: ALS 최적화 반복 횟수. default 15
- `use_cg`: Conjugate Gradient 사용 여부. default True
- `use_native`: Cython 최적화 사용 여부. default True
- `random state`: - 

$M\times N$ (유저 수 x 아이템 수) 형태의 `scipy.sparse.csr_matrix` 를 입력받는다. 해당 라이브러리는 내부적으로 관측된 셀의 값에 기본 신뢰도 `1`을 자동으로 더하여 계산하므로 ($c_{ui} = 1 + \text{data}$), 실제 구현 내에선 $d_{ui} = c_{ui} - 1$ 을 sparse matrix의 `data` 로 전달하여 수행한다. 

- `item_factors`
훈련 데이터의 각 아이템에 대한 잠재 벡터

- `user_factors`
훈련 데이터의 각 유저에 대한 잠재 벡터

- `fit(user_items, show_progress=True)`
유저-아이템 행령을 분해(Factorize)한다. 





