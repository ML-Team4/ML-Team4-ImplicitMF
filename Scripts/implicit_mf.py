from scipy.sparse import csr_matrix, csc_matrix
from pathlib import Path
import numpy as np
import pyarrow as pa
import csv
import json
import os
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENMP_NUM_THREADS"] = "8"
import implicit 
import time


class ImplicitMF:
    """
    Implicit-Matrix Factorization using Inverse Propensity Score, Mini-batch SGD and Negative Sampling.
    """

    def __init__(
        self,
        k_dim: int = 64,  # 유저/아이템 잠재 공간 차원 수 (Latent Vector Size, k)
        alpha: float = 40.0,  # 청취 횟수를 신뢰도(Confidence)로 변환하는 스케일링 계수 (Hu et al., 2008)
        gamma: float = 0.2, # IPS 하이퍼파라미터
        reg_weight: float = 0.01,  # L2 regularization weight
        iteration: int = 20,  # epochs
        random_state: int = 777,  
        top_k: int = 10, # the number of items to recommend
    ):
        self.k_dim = k_dim
        self.alpha = alpha
        self.gamma = gamma
        self.reg_weight = reg_weight
        self.iteration = iteration
        self.random_state = random_state
        self.top_k = top_k

        self.user_latent_matrix: np.ndarray | None = None
        self.item_latent_matrix: np.ndarray | None = None

        self.history: list[dict] = []

    def _initialize_latent_matrix(self, num_users: int, num_items: int):
        """_summary_
            행렬 가중치 초기화 -> N(0, 0.01^2) 
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

    def _parse_listener_counts(
        self, items: pa.Table | np.ndarray | None
    ) -> np.ndarray | None:
        """_summary_
            items.csv <= train_listener_count 추출
        """
        if items is None:
            return None
        if isinstance(items, np.ndarray):
            return items
        if (
            hasattr(items, "column_names")
            and "train_listener_count" in items.column_names
        ):
            return items["train_listener_count"].to_numpy()
        return np.asarray(items)

    def _parse_target_dict(
        self, targets: pa.Table | dict[int, set[int]]
    ) -> dict[int, set[int]]:
        """_summary_
            targets PyArrow 테이블을 {user_idx: set(item_idx)} dict 으로 변환
        """
        if isinstance(targets, dict):
            return targets

        target_u = targets["user_idx"].to_numpy()
        target_i = targets["item_idx"].to_numpy()

        target_dict = {}
        for u, i in zip(target_u, target_i):
            if u not in target_dict:
                target_dict[u] = set()
            target_dict[u].add(i)
        return target_dict

    def _update_users_als(self, D_csr: csr_matrix):
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
        """
        아이템 잠재 벡터 업데이트 (ALS Step + 1회 청취 아이템 Sherman-Morrison 고속 벡터화)
        """
        num_items = self.item_latent_matrix.shape[0]
        X = self.user_latent_matrix
        XTX_reg = X.T @ X + self.reg_weight * np.eye(self.k_dim, dtype=np.float32)

        A_inv = np.linalg.inv(XTX_reg)
        Z = X @ A_inv # shape: (num_users, k_dim)
        s = (X * Z).sum(axis=1) # shape: (num_users,)

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
            items (pa.Table | np.ndarray): dataset['items']
            val_data (tuple): dataset['evaluation']['validation'] tuple ((eligible,targets))
            patience (int): Early Stopping 기준
        """

        # 유저 잠재 행렬 (num_users x k_dim), 아이템 잠재 행렬 (num_items x k_dim) 초기화
        num_users, num_items = counts.shape
        self._initialize_latent_matrix(num_users, num_items)

        train_listener_counts = self._parse_listener_counts(items_dataset)
        val_users, val_target_dict = None, None
        if val_data is not None:
            val_users, val_targets_table = val_data
            val_target_dict = self._parse_target_dict(val_targets_table)

        # IPS = 1 / P_i 계산
        if self.gamma > 0.0 and train_listener_counts is not None:
            pop = train_listener_counts.astype(np.float32)
            max_pop = np.max(pop) if np.max(pop) > 0 else 1.0
            propensity = (pop / max_pop) ** self.gamma
            ips = (1.0 / np.clip(propensity, 1e-4, 1.0)).astype(np.float32)
        else:
            ips = np.ones(num_items, dtype=np.float32)

        # 행렬을 COO(Coordinate) 으로 변환해서 (user, item, listen_count) 1D 배열 추출
        coo = counts.tocoo()
        u_arr = coo.row
        i_arr = coo.col
        r_arr = coo.data.astype(np.float32)

        # user-item confidence 계산
        # c_ui = 1 + alpha * log(1 + listen_count) ... (np.log1p: log(1+x))
        # 희소 오차 잔차 행렬 d_ui = c_ui - 1 생성
        w_i = ips[i_arr]
        d_ui = (self.alpha * w_i * np.log1p(r_arr)).astype(np.float32)

        D_csr = csr_matrix((d_ui, (u_arr, i_arr)), shape=(num_users, num_items))
        D_csc = D_csr.tocsc()

        user_item_csr = csr_matrix((d_ui, (u_arr, i_arr)), shape=(num_users, num_items))

        als_engine = implicit.als.AlternatingLeastSquares(
            factors=self.k_dim,
            regularization=self.reg_weight,
            iterations=1,  
            random_state=self.random_state,
        )

        best_recall = -np.inf
        patience_count = 0
        best_user_emb = None
        best_item_emb = None
        self.history = []

        rng = np.random.default_rng(self.random_state)
        # ========================
        # 학습 시작
        # ========================
        print(f"[Implicit-MF] Train start.")
        for epoch in range(1, self.iteration + 1):
            t0 = time.time()

            #self._update_users_als(D_csr)
            #self._update_items_als(D_csc)
            als_engine.fit(user_item_csr, show_progress=False)

            t_fit = time.time() - t0

            # 4. C++ 엔진에서 계산된 Latent Matrix를 클래스 멤버 변수로 동기화
            self.user_latent_matrix = np.asarray(als_engine.user_factors)
            self.item_latent_matrix = np.asarray(als_engine.item_factors)

            log = {"epoch": epoch,}
            self.history.append(log)

            t0 = time.time()

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

                if metrics['recall@10'] > best_recall:
                    best_recall = metrics['recall@10']
                    patience_count = 0
                    best_user_emb = self.user_latent_matrix.copy()
                    best_item_emb = self.item_latent_matrix.copy()
                else:
                    patience_count += 1
                    if patience_count >= patience:
                        print(f"\n[Early Stopping]")
                        self.user_latent_matrix = best_user_emb
                        self.item_latent_matrix = best_item_emb
                        break
            else:
                print(f"Epoch {epoch:02d}/{self.iteration:02d} complete")

            

        return self

    def recommend(
        self,
        user_idx: int,
        train_seen: np.ndarray | set,
        top_k: int = 10,
    ) -> np.ndarray:
        
        if self.user_latent_matrix is None or self.item_latent_matrix is None:
            raise ValueError()

        # calculate scores: (k_dim,) @ (num_items, k_dim)^T -> (num_items,)
        u_emb = self.user_latent_matrix[user_idx]
        scores = u_emb @ self.item_latent_matrix.T

        # mask
        if len(train_seen) > 0:
            if isinstance(train_seen, set):
                scores[list(train_seen)] = -np.inf
            else:
                scores[train_seen] = -np.inf

        # Top-K 
        top_k_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_k_idx = top_k_idx[np.argsort(-scores[top_k_idx])]

        return top_k_idx

    def evaluate(
        self,
        eligible_users: np.ndarray,
        target_dict: dict[int, set[int]],
        train_counts: csr_matrix,
        train_listener_counts: np.ndarray,
        top_ks: list[int] = [10, 50, 100],
        user_batch_size: int = 32,
    ) -> tuple[float, float]:
        if self.user_latent_matrix is None or self.item_latent_matrix is None:
            raise ValueError()

        max_k = max(top_ks)
        recalls = {k: [] for k in top_ks}
        arps = []
        num_users = len(eligible_users)

        for start_idx in range(0, num_users, user_batch_size):
            end_idx = min(start_idx + user_batch_size, num_users)
            batch_users = eligible_users[start_idx:end_idx]

            scores_batch = (
                self.user_latent_matrix[batch_users] @ self.item_latent_matrix.T
            )

            for idx, u in enumerate(batch_users):
                truth = target_dict.get(u, set())
                if not truth:
                    continue

                train_seen = train_counts.getrow(u).indices
                scores = scores_batch[idx]

                if len(train_seen) > 0:
                    if isinstance(train_seen, set):
                        scores[list(train_seen)] = -np.inf
                    else:
                        scores[train_seen] = -np.inf

                top_max_idx = np.argpartition(scores, -max_k)[-max_k:]
                top_max_sorted = top_max_idx[np.argsort(-scores[top_max_idx])]

                for k in top_ks:
                    top_k_items = top_max_sorted[:k]
                    hits = len(set(top_k_items) & truth)
                    recalls[k].append(hits / len(truth))

                # Top-10 기준 ARP 수집
                arps.append(np.mean(train_listener_counts[top_max_sorted[:10]]))

        metrics = {
            f"recall@{k}": float(np.mean(recalls[k])) if recalls[k] else 0.0
            for k in top_ks
        }
        metrics["arp"] = float(np.mean(arps)) if arps else 0.0

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
                    self.learning_rate,
                    self.batch_size,
                    self.n_negatives,
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
            raise FileNotFoundError(
                f"no file exist: {filepath}"
            )

        data = np.load(filepath)
        self.user_latent_matrix = data["user_latent_matrix"]
        self.item_latent_matrix = data["item_latent_matrix"]

        config = data["config"]
        self.k_dim = int(config[0])
        self.alpha = float(config[1])
        self.gamma = float(config[2])
        self.reg_weight = float(config[3])
        self.learning_rate = float(config[4])
        self.batch_size = int(config[5])
        self.n_negatives = int(config[6])
        self.iteration = int(config[7])
        self.top_k = int(config[8])

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
