# Train Implcit-MF based ALS with IPS

> - Implicit-MF & ALS optimization
> [Hu, Y., Koren, Y., & Volinsky, C. (2008, December). Collaborative filtering for implicit feedback datasets. In 2008 Eighth IEEE international conference on data mining (pp. 263-272). Ieee.](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4781121)
> - IPS framework
> [Schnabel, Tobias, et al. "Recommendations as treatments: Debiasing learning and evaluation." international conference on machine learning. PMLR, 2016.](https://proceedings.mlr.press/v48/schnabel16.pdf)
> - IPS for Implicit feedback 
> [Saito, Yuta, et al. "Unbiased recommender learning from missing-not-at-random implicit feedback." Proceedings of the 13th international conference on web search and data mining. 2020.](https://dl.acm.org/doi/pdf/10.1145/3336191.3371783)
> 

여기선 Implicit Feedback (청취횟수) 기반의 추천시스템 Implicit Matrix factorization (Implicit ALS)의 학습 메커니즘, 추천/평가 연산 구조를 설명한다. 

## Data Formulation
음악 청취 횟수와 같은 implicit feedback data 는 explicit feedback data (e.g., rating) 과 달리, 부정적 선호의 정보가 명확하지 않다. 따라서 청취 여부에 따른 선호도와 청취 횟수에 따른 신뢰도로 분리하여 정의한다. 

### 선호도
사용자 $u$ 가 아이템 $i$ 를 한번이라도 청취했는지 여부를 다음과 같이 나타낸다. 

$$
p_{ui} = \begin{cases}
    1 & \text{if } r_{ui} > 0\\
    0 & \text{if } r_{ui} = 0
\end{cases} \quad (r_{ui} : \text{사용자 $u$ 의 아이템 $i$ 에 대한 관측된 청취 횟수})
$$

### 신뢰도 
청취 횟수 $r_{ui}$ 가 높을수록 사용자 $u$ 가 아이템 $i$ 를 실제로 선호할 확률이 높다는 신뢰도를 다음과 같이 정의한다. 

$$
c_{ui} = 1 + \alpha \cdot r_{ui}
$$

- $\alpha \geq 0$: 신뢰도 제어 하이퍼파라미터
 
이는 미청취 아이템 ($r_{ui} = 0$) 에 대해서 기본 신뢰도 $c_{ui} = 1$ 을 가진다. 

<details>
<summary>Code</summary>

코드 내에선 안정성을 위해 $r_{ui}$ 에 로그를 적용하여 수행한다. 이때 $r_{ui} = 0$ 인 경우의 수치적 안정성을 위해 다음과 같이 구현한다. 

``` python
c_ui = 1 + alpha * np.log(1 + r_ui)
```
</details>

## Loss function (Implicit-MF)
Implicit MF 는 사용자 잠재 벡터 $x_u \in \mathbb{R}^{k}$, 아이템 잠재 벡터 $y_i \in \mathbb{R}^{k}$ 에 대해, 내적 $u^{T}i$ 가 선호도 $p_{ui}$ 를 근사하도록 학습되며, 손실 함수에 신뢰도 $c_{ui}$ 에 따른 가중치와 $L_2$ 정규화를 적용한다. 

$$
\mathcal{L} = \sum_{u=1}^{M} \sum_{i=1}^{N} c_{ui} \left( p_{ui} - x_u^T y_i \right)^2 + \lambda \left( \sum_{u=1}^{M} \Vert{}x_u\Vert{}_2^2 + \sum_{i=1}^{N} \Vert{}y_i\Vert{}_2^2 \right)
$$

- $\lambda \geq 0$: $L_2$ 가중치 하이퍼파라미터 
- $M,N$ : 유저, 아이템 수
- $k$ : 각 유저, 아이템에 할당할 잠재 차원 

> 최적화는 ALS 기반으로 진행하므로, 실제 최적화 과정에서는 손실값을 따로 계산하지 않는다. 

### apply IPS(Inverse Propensity Score)
추천시스템에서의 로그 데이터에는 사용자가 무작위로 아이템을 소비한 결과가 아닌, 노출 빈도가 높은 인기곡 위주로 소비하는 Selection bias 가 발생한다. (타인의 추천, 유행 등과 같은 사회적 영향이 유저의 아이템 소비에 영향을 줄 수 있음 -> 로그에 기록된 유저가 소비한 아이템들이 정말로 유저가 선호해서 한 것이 아닌, 그저 인기가 많아서 소비 가능)  
이러한 편향에 대해 보정 없이 학습하면 유저에 대해 인기 아이템만 반복적으로 추천하는 Population bias 가 발생/심화 된다.  

이를 위해 Inverse Propensity Score, IPS 를 손실함수에 결합하여, 각 관측 상호작용의 오차에 관측 확률(Propensity Score)을 역으로 가중치를 부여한다. 

즉, 관측이 적은 비인기 아이템에 대해 전체 손실의 비중을 키우고, 관측이 많은 인기 아이템에 대해 전체 손실에 차지하는 비중을 줄인다.  

#### Propensity Score
학습 데이터에서 아이템 $i$ 가 관측될 확률 (Propensity Score) 은 아이템별 고유 청취자 수 `train_listener_counts` 에 비례한다 가정하여 다음과 같이 계산한다. 

$$
P_i = \left( \frac{\text{Pop}(i)}{\max_j \text{Pop}(j)} \right)^\gamma \quad (\gamma \in [0.0, 1.0])
$$

- $\text{Pop}(i)$ : 학습 기간 동안 아이템 $i$ 와 상호작용한(청취한) 고유 사용자 수
- $\gamma$ : 편향 보정 강도 조절 하이퍼파라미터, $\gamma = 0$ 인 경우, 편향 보정을 적용하지 않는다. (기존 Implicit MF와 동일)

<details>
<summary>Code</summary>
아이템별 고유 청취자 수 `train_listener_counts` 에 기반하여, 다음과 같이 구현한다.

``` python
pop = train_listener_counts.astype(np.float32)
max_pop = np.max(pop) if np.max(pop) > 0 else 1.0
propensity = (pop / max_pop) ** self.gamma
```
</details>

#### Loss function applied IPS 
앞서 계산한 Propensity Score 는 기존 손실함수에 역 가중치 형태로 다음과 같이 포함된다.

$$
c_{ui}^{\text{IPS}} = \begin{cases}
    \frac{c_{ui}}{P_i} & \text{if } r_{ui} > 0 \\
    1 & \text{if } r_{ui} = 0
\end{cases}
$$

$$
\mathcal{L} = \sum_{u=1}^{M} \sum_{i=1}^{N} c_{ui}^{\text{IPS}} \left( p_{ui} - x_u^T y_i \right)^2 + \lambda \left( \sum_{u=1}^{M} \Vert{}x_u\Vert{}_2^2 + \sum_{i=1}^{N} \Vert{}y_i\Vert{}_2^2 \right)
$$

<details>
<summary>Code</summary>
실제 코드 상에선, 이후 Sparse matrix 연산을 위해 다음과 같이 $d_{ui} = c_{ui}^{IPS} - 1$ 로 치환하여 계산한다. 

```python
w_i = 1.0 /propensity
d_ui = alpha * w_i * np.log1p(r_ui)

```
</details>

## Optimization

WALS(Weighted Alternating Least Squares) 를 사용하여 최적화를 진행한다. 

학습 시작 전 사용자 잠재 행렬, 아이템 잠재 행렬은 `np.random.default_rng(random_state)`를 통해 시드를 고정한 뒤, 각 가중치를 정규분포 $\mathcal{N}(0, \frac{1}{k})$ 로 초기화하여 수행한다.


### 사용자 잠재 벡터

전체 아이템 행렬 $Y \in \mathbb{R}^{N \times k}$ 를 고정한 채, 유저 잠재 행렬에 대해 다음과 같이 계산한다. 

$$
x_u = (Y^TC_uY + \lambda \cdot I)^{-1}Y^TC_up_u
$$

- $C_u \in \mathbb{R}^{N \times N}$ : 대각 성분이 $c_{ui}$ 인 대각행렬
- $p_u \in \mathbb{R}^N$ : 사용자 $u$의 모든 아이템에 대한 선호도 벡터
- $I \in \mathbb{R}^{k \times k}$

#### apply IPS 
기존 $C_u$ 에 대해 다음과 같이 치환하여 최적화를 진행한다. 

$$
x_u = \left( Y^T C_u^{\text{IPS}} Y + \lambda I \right)^{-1} Y^T C_u^{\text{IPS}} p_u
$$

<details>
<summary>Code</summary>
$x_u = A^{-1}b$ 꼴이므로, 선형방정식 $Ax_u = b$ 을 계산하여 구한다. (표기 간편을 위해 $C_u = C_u^{\text{IPS}}$)

$A = Y^T C_u Y + \lambda I$ 에 대해, $C_u = I_N + D_u$ ($d_ui = c_ui - 1$) 로 분해하여 전개한다. (기존 $C_u$ 를 그대로 사용할 시, 전체 유저-아이템 에 대해 연상량이 급증하므로, 이와 같이 분해하여 계산한다.)

$$
\begin{align}
    A &= Y^T(I_N + D_u)Y + \lambda \cdot I\\
    &= Y^TY + Y^TD_uY + \lambda \cdot I
\end{align}
$$

```python
Y = self.item_latent_matrix
YTY = Y.T @ Y
I_k = np.eye(k)

A = YTY + (Y_u * d_ui).T @ Y_u + alpha * I_k
```

이어서 $b = Y^TC_up_u$ 에 대해, $p_u$는 미청취 곡에 대해서 0이므로, 다음과 같이 전개할 수 있다.
$$
b = Y^TC_up_u = \sum_i c_{ui}p_{ui}y_i = \sum_i(1+d_{ui})y_i
$$

```python
b = (Y_u * (1.0 + d_ui)).sum(axis=0)
```

이후 `np.linalg.solve(A, b)` 를 통해 $x_u$를 계산한다.

```python
x_u = np.linalg.solve(A, b)
```

</details>

### 아이템 잠재 벡터 
전체 사용자 행렬 $X \in \mathbb{R}^{M \times k}$ 를 고정한 채, 아이템 잠재 행렬에 대해 다음과 같이 계산한다. 

$$
y_i = (X^TC_iX + \lambda \cdot I)^{-1}X^TC_ip_i
$$

- $C_i \in \mathbb{R}^{M \times M}$ : 대각 성분이 $c_{ui}$ 인 대각행렬
- $p_i \in \mathbb{R}^M$ : 아이템 $i$ 에 대한 모든 사용자의 선호도 벡터

#### apply IPS
마찬가지로 기존 $C_i$ 에 대해 다음과 같이 치환하여 최적화를 진행한다.

$$
y_i = (X^TC_i^{\text{IPS}}X + \lambda \cdot I)^{-1}X^TC_i^{\text{IPS}}p_i
$$

코드는 앞선 사용자 잠재 벡터에서의 실제 코드 구현과 거의 동일하므로 생략한다. 

## 추론 (Top-K Recommendation)
학습된 유저, 아이템 잠재 행렬에 대해, 내적 ($x_u^Ty_i$) 를 통해 Score 를 계산하고, 이를 기반으로 Top-K를 추출한다.

### Score 계산
$$
\text{score} = Y^Tx_u \in \mathbb{R}^N
$$

### Masking
학습에 사용된 데이터셋을 이전에 유저가 청취한 음악으로 간주하고, 유저가 한번도 듣지 않은 음악에 국한하여 Top-K 추천을 진행한다. 

$$
\text{score}_{ui} \leftarrow -\infty\quad(\forall i \in I_u^0)
$$

- $I_u^0$ : 학습 데이터 내 유저 $u$ 에 대해 상호작용이 관측된(청취한) 아이템의 집합

### Top-K
계산된 $\text{score}$ 에 기반하여, 상위 $K$ 개의 아이템을 추출한다. 

## 평가 

다음 두 metric 을 통해 성능을 평가한다. 

$$
\mathrm{Recall}@K = \frac{1}{\vert\mathcal{U}\vert}\sum_{u \in \mathcal{U}} \frac{\vert R_u^{(K)}\cap T_u \vert}{\vert T_u\vert} 
$$

$$
\mathrm{ARP} = \frac{1}{\vert\mathcal{U}\vert}\sum_{u \in \mathcal{U}} \left(\frac{1}{K}\sum_{i \in R_u^{(K)}}\mathrm{Pop}(i)\right)
$$

- $\mathcal{U}~(\vert\mathcal{U}\vert = M')$ : 평가에 사용될 데이터셋에서의 유저 인덱스 집합 
- $R_u^{(K)}$ : 유저 $u$ 에 대해 추천된 $K$ 개의 아이템 집합
- $T_u$ : 유저 $u$ 에 대한 Ground Truth 아이템 집합

두 지표 모두 모든 유저에 대해 평균을 내어 산출한다. 
