import os
import sys
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENMP_NUM_THREADS"] = "8"

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.utils import parse_target_dict, parse_listener_counts

from scipy.sparse import csr_matrix, csc_matrix
import numpy as np
import pyarrow as pa
import csv
import json
import os
# OpenBLAS, MKL, OpenMP 라이브러리에서 사용하는 스레드 수를 제한하여 CPU 사용량을 조절하고, 학습 속도를 향상시키기 위해 설정

import implicit 
import time
import urllib.request
import configs

class ImplicitMF:
    """
    Implicit-Matrix Factorization using Inverse Propensity Score, Mini-batch SGD and Negative Sampling.
    """
    def __init__(
        self,
        k_dim: int = 64,  # 유저/아이템 잠재 공간 차원 수 (Latent Vector Size, k)
        alpha: float = 40.0,  # 청취 횟수를 Confidence로 변환하는 스케일링 계수 (Hu et al., 2008)
        gamma: float = 0.2, # IPS 하이퍼파라미터
        reg_weight: float = 0.01,  # L2 regularization weight
        iteration: int = 20,  # epochs
        random_state: int = 777,  
        top_k: int = 10, # the number of items to recommend
        use_implicit_als: bool = True, # implicit 라이브러리 ALS 사용 여부
    ):
        self.k_dim = k_dim
        self.alpha = alpha
        self.gamma = gamma
        self.reg_weight = reg_weight
        self.iteration = iteration
        self.random_state = random_state
        self.top_k = top_k
        self.use_implicit_als = use_implicit_als

        self.user_latent_matrix: np.ndarray | None = None
        self.item_latent_matrix: np.ndarray | None = None

        self.history: list[dict] = []

    def _initialize_latent_matrix(self, num_users: int, num_items: int):
        """_summary_
            (num_users x k_dim), (num_items x k_dim) 행렬 가중치 초기화 -> N(0, 1.0/k_dim) 
        Args:
            num_users (int): number of users
            num_items (int): number of items 
        """
        rng = np.random.default_rng(self.random_state)

        scale = 1.0 / np.sqrt(self.k_dim)
        
        self.user_latent_matrix = rng.normal(
            scale=scale, size=(num_users, self.k_dim)
        ).astype(np.float32)
        self.item_latent_matrix = rng.normal(
            scale=scale, size=(num_items, self.k_dim)
        ).astype(np.float32)

    

    def _update_users_als(self, D_csr: csr_matrix):
        """_summary_
            ALS 방식으로 유저 잠재 행렬 업데이트
        Args:
            D_csr (csr_matrix): user-item sparse matrix with d_ui
        """
        num_users = self.user_latent_matrix.shape[0]
        Y = self.item_latent_matrix
        YTY = Y.T @ Y
        I_k = np.eye(self.k_dim, dtype=np.float32)

        for u in range(num_users):
            start_idx = D_csr.indptr[u]
            end_idx = D_csr.indptr[u+1]
            if start_idx == end_idx:
                continue

            i_nodes = D_csr.indices[start_idx:end_idx]
            d_u = D_csr.data[start_idx:end_idx]

            Y_u = Y[i_nodes]
            # A = YTY + Y_u^T diag(d_u) Y_u + reg * I
            A = YTY + (Y_u * d_u[:, None]).T @ Y_u + self.reg_weight * I_k
            # b = sum((1 + d_ui) * y_i)
            b = (Y_u * (1.0 + d_u)[:, None]).sum(axis=0)

            self.user_latent_matrix[u] = np.linalg.solve(A, b)

    def _update_items_als(self, D_csc: csc_matrix):
        X = self.user_latent_matrix
        XTX_reg = X.T @ X + self.reg_weight * np.eye(self.k_dim, dtype=np.float32)

        A_inv = np.linalg.inv(XTX_reg)
        Z = X @ A_inv 
        s = (X * Z).sum(axis=1) 

        item_degrees = np.diff(D_csc.indptr)

        items_0 = np.where(item_degrees == 0)[0]
        if len(items_0) > 0:
            self.item_latent_matrix[items_0] = 0.0

        items_1 = np.where(item_degrees == 1)[0]
        if len(items_1) > 0:
            u_nodes_1 = D_csc.indices[D_csc.indptr[items_1]]
            d_vals_1 = D_csc.data[D_csc.indptr[items_1]]
            weights_1 = (1.0 + d_vals_1) / (1.0 + d_vals_1 * s[u_nodes_1])
            self.item_latent_matrix[items_1] = Z[u_nodes_1] * weights_1[:, None]

        items_multi = np.where(item_degrees > 1)[0]
        for i in items_multi:
            start_idx = D_csc.indptr[i]
            end_idx = D_csc.indptr[i+1]
            u_nodes = D_csc.indices[start_idx:end_idx]
            d_vals = D_csc.data[start_idx:end_idx]

            X_i = X[u_nodes]
            A = XTX_reg + (X_i * d_vals[:, None]).T @ X_i
            b = (X_i * (1.0 + d_vals)[:, None]).sum(axis=0)

            self.item_latent_matrix[i] = np.linalg.solve(A, b)

    def train(
        self,
        counts: csr_matrix,
        items_dataset: pa.Table | np.ndarray | None = None,
        val_data: tuple[np.ndarray, pa.Table] | None = None,
        patience: int = 5,
    ):
        """_summary_
            Train implicit-MF 
        Args:
            counts (csr_matrix): user-item matrix
            items_dataset (pa.Table | np.ndarray | None): dataset['items']
            val_data (tuple): dataset['evaluation']['validation'] tuple ((eligible,targets))
            patience (int): Early Stopping 기준
        """

        # 유저 잠재 행렬 (num_users x k_dim), 아이템 잠재 행렬 (num_items x k_dim) 초기화
        num_users, num_items = counts.shape
        self._initialize_latent_matrix(num_users, num_items)

        # validation set parsing
        train_listener_counts = parse_listener_counts(items_dataset)
        val_users, val_target_dict = None, None
        if val_data is not None:
            val_users, val_targets_table = val_data
            val_target_dict = parse_target_dict(val_targets_table)

        # 행렬을 COO(Coordinate) 로 변환해서 user, item, listen_count 1D 배열 추출
        coo = counts.tocoo()
        u_arr = coo.row
        i_arr = coo.col
        r_arr = coo.data.astype(np.float32)

        # IPS = 1 / P_i 계산
        if self.gamma > 0.0 and train_listener_counts is not None:
            pop = train_listener_counts.astype(np.float32)
            max_pop = np.max(pop) if np.max(pop) > 0 else 1.0
            propensity = (pop / max_pop) ** self.gamma
            ips = (1.0 / np.clip(propensity, 1e-4, 1.0)).astype(np.float32)
        else:
            ips = np.ones(num_items, dtype=np.float32)

        # w_i = IPS (Inverse Propensity Score)
        # d_ui = alpha * w_i * log(1 + r_ui) 
        w_i = ips[i_arr]
        d_ui = (self.alpha * w_i * np.log1p(r_arr)).astype(np.float32)

        # D_csr: (num_users x num_items) sparse matrix with d_ui
        # D_csc: (num_items x num_users) sparse matrix with d_ui
        # user_item_csr: (num_users x num_items) sparse matrix with d_ui
        if self.use_implicit_als:
            D_csr = csr_matrix((d_ui, (u_arr, i_arr)), shape=(num_users, num_items))
            D_csc = D_csr.tocsc()
            user_item_csr = csr_matrix((d_ui, (u_arr, i_arr)), shape=(num_users, num_items))
            als_engine = implicit.als.AlternatingLeastSquares(
                factors=self.k_dim,
                regularization=self.reg_weight,
                iterations=1,  
                random_state=self.random_state,
            )

        # Early Stopping 변수 초기화
        # validation set 에 대한 Recall@10 기준으로 가장 높은 성능을 기록한 epoch 이후 patience 횟수만큼 성능 향상이 없으면 학습 종료
        best_recall = -np.inf
        patience_count = 0
        best_user_emb = None
        best_item_emb = None

        # 학습 로그 
        self.history = []

        # ========================
        # 학습 시작
        # ========================
        print(f"[Implicit-MF] Train start.")
        for epoch in range(1, self.iteration + 1):
            t0 = time.time()

            # ALS 방식으로 유저/아이템 잠재 행렬 업데이트
            if self.use_implicit_als:
                als_engine.fit(user_item_csr, show_progress=False)
                self.user_latent_matrix = np.asarray(als_engine.user_factors)
                self.item_latent_matrix = np.asarray(als_engine.item_factors)
            else:
                self._update_users_als(D_csr)
                self._update_items_als(D_csc)

            # t_fit: ALS 학습 소요 시간
            t_fit = time.time() - t0
            
            log = {"epoch": epoch}
            self.history.append(log)

            t0 = time.time()

            # validation set 에 대한 평가 수행 (5 epoch 마다 또는 마지막 epoch)
            if val_users is not None and val_target_dict is not None and (
                epoch % 5 == 0 or epoch == self.iteration
            ):
                metrics = self.evaluate(
                    eligible_users=val_users,
                    target_dict=val_target_dict,
                    train_counts=counts,
                    train_listener_counts=train_listener_counts,
                    top_ks=[self.top_k, 50, 100]
                )

                t_eval = time.time() - t0

                log.update(metrics)

                print(
                    f"Epoch {epoch:02d}/{self.iteration:02d} | "
                    f"R@10: {metrics['recall@10']:.6f} | R@50: {metrics['recall@50']:.6f} | "
                    f"R@100: {metrics['recall@100']:.6f} | ARP: {metrics['arp']:.2f} | "
                    f"ALS Fit: {t_fit:.2f}s | Evaluate: {t_eval:.2f}s"
                )

                # Early Stopping
                if metrics['recall@10'] > best_recall:
                    best_recall = metrics['recall@10']
                    patience_count = 0
                    best_user_emb = self.user_latent_matrix.copy()
                    best_item_emb = self.item_latent_matrix.copy()
                else:
                    patience_count += 1
                    if patience_count >= patience:
                        print(f"\n[Early Stopping]")
                        break
            else:
                print(f"Epoch {epoch:02d}/{self.iteration:02d} complete")

        if best_user_emb is not None and best_item_emb is not None:
            self.user_latent_matrix = best_user_emb
            self.item_latent_matrix = best_item_emb
        
        return self

    def recommend(
        self,
        user_idx: int | np.ndarray | list[int],
        train_seen: np.ndarray | set | list | csr_matrix | None = None,
        top_k: int = 10,
    ) -> np.ndarray:
        """_summary_
            유저, 아이템 잠재 행렬을 통해 Score 계산 후 Top-K 추천 아이템 추출
        Args:
            user_idx (int | np.ndarray | list[int]): 추천 대상 유저 인덱스
            train_seen (np.ndarray | set | list | csr_matrix | None): 학습 데이터에서 유저가 이미 상호작용한 아이템 인덱스
            top_k (int, optional): 추천할 아이템 수. Defaults to 10.

        Returns:
            np.ndarray: 추천 아이템 인덱스 (Top-K)
        """
        if self.user_latent_matrix is None or self.item_latent_matrix is None:
            raise ValueError()

        u_mat = np.asarray(self.user_latent_matrix, dtype=np.float32)
        i_mat = np.asarray(self.item_latent_matrix, dtype=np.float32)

        is_single = isinstance(user_idx, (int, np.integer))

        if is_single:
            # calculate scores: (k_dim,) @ (num_items, k_dim)^T -> (num_items,)
            # 개별 유저를 대상으로 진행
            scores = u_mat[user_idx] @ i_mat.T
        
            # 학습 데이터에서 이미 상호작용한 아이템은 추천 후보에서 제외 -> score를 -inf로 설정
            if train_seen is not None and len(train_seen) > 0:
                seen_idx = (
                    list(train_seen) if isinstance(train_seen, set) else train_seen
                )
                scores[seen_idx] = -np.inf
        
            # Top-K 
            # np.argpartition: scores 배열에서 상위 top_k 개의 인덱스 탐색
            # np.argsort: 상위 top_k 개의 인덱스를 score 기준으로 내림차순으로 정렬
            top_k_idx = np.argpartition(scores, -top_k)[-top_k:]
            top_k_idx = top_k_idx[np.argsort(-scores[top_k_idx])]
            return top_k_idx
        else:
            # (batch_size, k_dim) @ (num_items, k_dim)^T -> (batch_size, num_items)
            batch_users = np.asarray(user_idx)
            scores_batch = u_mat[batch_users] @ i_mat.T

            # 학습 데이터에서 이미 상호작용한 아이템은 추천 후보에서 제외 -> score를 -inf로 설정
            if train_seen is not None:
                if isinstance(train_seen, csr_matrix):
                    sub_csr = train_seen[batch_users]
                    row_lengths = np.diff(sub_csr.indptr)
                    seen_rows = np.repeat(np.arange(len(batch_users)), row_lengths)
                    scores_batch[seen_rows, sub_csr.indices] = -np.inf
                elif isinstance(train_seen, (list, tuple)):
                    for idx, seen in enumerate(train_seen):
                        if len(seen) > 0:
                            seen_idx = list(seen) if isinstance(seen, set) else seen
                            scores_batch[idx, seen_idx] = -np.inf

            # Top-K
            # 위와 동일
            top_k_idx = np.argpartition(scores_batch, -top_k, axis=-1)[:, -top_k:]
            row_indices = np.arange(len(batch_users))[:, None]
            top_scores = scores_batch[row_indices, top_k_idx]

            # np.take_along_axis: top_k_idx 배열을 top_scores 기준으로 정렬하여 추천 아이템 인덱스 반환
            sort_order = np.argsort(-top_scores, axis=-1)
            return np.take_along_axis(top_k_idx, sort_order, axis=-1)

    def evaluate(
        self,
        eligible_users: np.ndarray,
        target_dict: dict[int, set[int]],
        train_counts: csr_matrix,
        train_listener_counts: np.ndarray,
        top_ks: list[int] = [10, 50, 100],
        user_batch_size: int = 256,
    ) -> tuple[float, float]:
        """_summary_
        1. eligible_users: 평가 대상 유저 인덱스 배열
        Args:
            eligible_users (np.ndarray): 평가 대상 유저 인덱스 배열
            target_dict (dict[int, set[int]]): 평가 대상 유저별 ground truth 아이템 인덱스 집합
            train_counts (csr_matrix): 학습 데이터 user-item sparse matrix
            train_listener_counts (np.ndarray): 학습 데이터 아이템별 청취자 수 배열
            top_ks (list[int], optional): 추천할 아이템 수 목록. Defaults to [10, 50, 100].
            user_batch_size (int, optional): 유저 배치 크기. Defaults to 32.

        Returns:
            tuple[float, float]: Recall@K, ARP
        """
        if self.user_latent_matrix is None or self.item_latent_matrix is None:
            raise ValueError()

        valid_users = np.array(
            [u for u in eligible_users if target_dict.get(int(u))],
            dtype=eligible_users.dtype,
        )
        num_users = len(valid_users)
        if num_users == 0:
            metrics = {f"recall@{k}": 0.0 for k in top_ks}
            metrics["arp"] = 0.0
            return metrics
        
        max_k = max(top_ks)
        recalls = {k: [] for k in top_ks}
        batch_arps = []
        
        # 유저 배치 단위로 추천 수행
        for start_idx in range(0, num_users, user_batch_size):
            end_idx = min(start_idx + user_batch_size, num_users)
            batch_users = eligible_users[start_idx:end_idx]

            batch_top_items = self.recommend(
                user_idx=batch_users, train_seen=train_counts, top_k=max_k
            )

            batch_arps.append(
                train_listener_counts[batch_top_items[:, :10]].mean(axis=1)
            )

            for idx, u in enumerate(batch_users):
                truth = target_dict[int(u)]
                top_max_sorted = batch_top_items[idx]
                cum_hits = np.cumsum([item in truth for item in top_max_sorted])
                n_truth = len(truth)

                for k in top_ks:
                    recalls[k].append(cum_hits[k - 1] / n_truth)

        metrics = {
            f"recall@{k}": float(np.mean(recalls[k])) if recalls[k] else 0.0
            for k in top_ks
        }
        metrics["arp"] = float(np.concatenate(batch_arps).mean())

        return metrics
    
    def save_history(self, filepath: str | Path):
        """_summary_
            학습 로그 저장
        Args:
            filepath (str | Path): 저장 경로
        """
        if not self.history:
            print("[ImplicitMF] No log")
            return

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if filepath.suffix.lower() == ".csv":
            keys = self.history[0].keys()
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.history)
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=4, ensure_ascii=False)

        print(f"[ImplicitMF] Success to save history: {filepath}")

    def save_model(self, filepath: str | Path):
        """_summary_
            모델 파라미터, 하이퍼파라미터 저장 (.npz)
        Args:
            filepath (str | Path): 저장경로

        """
        if self.user_latent_matrix is None or self.item_latent_matrix is None:
            raise ValueError()

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            filepath,
            user_latent_matrix=self.user_latent_matrix,
            item_latent_matrix=self.item_latent_matrix,
            config=np.array(
                [
                    self.k_dim,
                    self.alpha,
                    self.gamma,
                    self.reg_weight,
                    self.iteration,
                    self.top_k,

                ]
            ),
        )
        print(f"[ImplicitMF] Success to save model: {filepath}")

    def load_model(self, filepath: str | Path):
        """_summary_
            .npz 로부터 load
        Args:
            filepath (str | Path): path to .npz

        """
        filepath = Path(filepath)
        if not filepath.exists():
            filepath.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {filepath.name}")
            urllib.request.urlretrieve(f"{configs.RELEASE_URL}/{filepath.name}", filepath)
        
        data = np.load(filepath)
        self.user_latent_matrix = data["user_latent_matrix"]
        self.item_latent_matrix = data["item_latent_matrix"]

        config = data["config"]
        self.k_dim = int(config[0])
        self.alpha = float(config[1])
        self.gamma = float(config[2])
        self.reg_weight = float(config[3])
        self.iteration = int(config[4])
        self.top_k = int(config[5])

        print(
            f"[ImplicitMF] Success to load: U{self.user_latent_matrix.shape}, I{self.item_latent_matrix.shape}"
        )
    
if __name__ == "__main__":
    from load_datasets import load_data
    
    datasets = load_data()

    model = ImplicitMF(
        k_dim=128,
        alpha=20.0,
        gamma=0.0,
        reg_weight=0.01,
        iteration=20,
        random_state=777,
    )

    model.train(
        counts=datasets["counts"],
        items_dataset=datasets["items"],
        val_data=datasets["evaluation"]["validation"], 
        patience=5,
    )
    model.save_model("checkpoints/implicit_mf_best.npz")
    model.save_history("logs/implicit_mf_history.json")
