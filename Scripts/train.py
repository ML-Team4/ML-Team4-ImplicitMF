import os
import sys
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENMP_NUM_THREADS"] = "8"

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from implicit_mf import ImplicitMF
import csv
import json
from itertools import product
from utils.utils import prepare_for_train
import argparse

import configs

def train_single(
    k_dim: int = 64,
    alpha: float = 40.0,
    gamma: float = 0.0,
    reg_weight: float = 0.01,
    iteration: int = 20,
    patience: int = 5,
    random_state: int = 777,
    top_k: int = 10,
    save_path: str | Path | None = None,
):
    counts, items, val_data = prepare_for_train()

    model = ImplicitMF(
        k_dim=k_dim,
        alpha=alpha,
        gamma=gamma,
        reg_weight=reg_weight,
        iteration=iteration,
        random_state=random_state,
        top_k=top_k,
    )
    model.train(
        counts=counts,
        items_dataset=items,
        val_data=val_data,
        patience=patience,
    )

    best_epoch_log = max(model.history, key=lambda x: x.get("recall@10", 0.0))
    print(
        f"[Single Train Complete] Best Epoch: {best_epoch_log['epoch']} | "
        f"R@10: {best_epoch_log.get('recall@10', 0.0):.6f} | "
        f"R@50: {best_epoch_log.get('recall@50', 0.0):.6f} | "
        f"R@100: {best_epoch_log.get('recall@100', 0.0):.6f} | "
        f"ARP: {best_epoch_log.get('arp', 0.0):.2f}"
    )

    target_path = (
        Path(save_path)
        if save_path
        else configs.CHECKPOINT_DIR / f"implicit_mf_k{k_dim}_a{alpha}_g{gamma}.npz"
    )
    model.save_model(target_path)
    print(f"Model saved to: {target_path}")

    return model, best_epoch_log

def grid_serach():
    """_summary_
    Grid Search를 수행하여 ImplicitMF 모델의 최적 하이퍼파라미터를 탐색 (validation set 기준)
    """
    counts, items, val_data = prepare_for_train()

    param_grid = {
        "k_dim": [64, 128, 256],         
        "alpha": [10.0, 20.0, 40.0],     
        "gamma": [0.0],   # baseline 에서는 0.0으로 고정, 이후 최적의 baseline 찾은 후에 gamma 변화별 편향 조정 여부 확인
        "reg_weight": [0.001, 0.01, 0.05], 
        "iteration": [20],               
        "random_state": [777],
        "top_k": [10],
    }

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = [dict(zip(keys, v)) for v in product(*values)]

    results = []
    best_recall_10 = -100.0
    best_config = None

    print(f"Grid Search Start")
    for idx, config in enumerate(combinations, 1):
        print(f"[{idx}/{len(combinations)}] Running Config: {config}")

        model = ImplicitMF(**config)
        model.train(
            counts=counts,
            items_dataset=items,
            val_data=val_data,
            patience=5,
        )

        best_epoch_log = max(model.history, key=lambda x: x.get("recall@10", 0.0))

        res = {
            **config,
            "best_epoch": best_epoch_log["epoch"],
            "best_val_recall@10": best_epoch_log.get("recall@10", 0.0),
            "best_val_recall@50": best_epoch_log.get("recall@50", 0.0),
            "best_val_recall@100": best_epoch_log.get("recall@100", 0.0),
            "best_val_arp": best_epoch_log.get("arp", 0.0),
        }
        results.append(res)

        print(
            f"Best:\n"
            f"R@10: {res['best_val_recall@10']:.6f} | "
            f"R@50: {res['best_val_recall@50']:.6f} | "
            f"R@100: {res['best_val_recall@100']:.6f}"
        )

        # 각 세팅에 대해 best Recall@10 가중치 저장
        if res["best_val_recall@10"] > best_recall_10:
            best_recall_10 = res["best_val_recall@10"]
            best_config = config
            model.save_model(configs.BASELINE_MODEL_DIR)

    with open(configs.LOG_DIR / "grid_search_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    if results:
        fieldnames = list(results[0].keys())
        with open(configs.LOG_DIR / "grid_search_results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print("===============================================================")
    print(f"Best R@10: {best_recall_10:.6f}")
    print(f"Best Config: {best_config}")
    print("===============================================================")

def compare_with_diff_gamma():
    """
    Grid Search로 찾은 최적 하이퍼파라미터를 고정한 상태에서
    gamma 값만 변경하며 validation set 기준으로 편향 완화(ARP) 및 성능(Recall)을 비교
    """
    counts, items, val_data = prepare_for_train()

    base_model = ImplicitMF()
    base_model.load_model(configs.BASELINE_MODEL_DIR)

    print(
        f"Loaded Best Baseline:\n"
        f"k_dim={base_model.k_dim} | alpha={base_model.alpha} | "
        f"gamma={base_model.gamma} | reg_weight={base_model.reg_weight}\n"
    )

    param_grid = {
        "k_dim": [base_model.k_dim],
        "alpha": [base_model.alpha],
        "gamma": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "reg_weight": [base_model.reg_weight],
        "iteration": [base_model.iteration],
        "random_state": [777],
        "top_k": [base_model.top_k],
    }

    del base_model

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = [dict(zip(keys, v)) for v in product(*values)]

    results = []

    print("Gamma Comparison Start")
    for idx, config in enumerate(combinations, 1):
        print(f"[{idx}/{len(combinations)}] Running Config: {config}")

        model = ImplicitMF(**config)
        model.train(
            counts=counts,
            items_dataset=items,
            val_data=val_data,
            patience=5,
        )

        best_epoch_log = max(model.history, key=lambda x: x.get("recall@10", 0.0))

        res = {
            **config,
            "best_epoch": best_epoch_log["epoch"],
            "best_val_recall@10": best_epoch_log.get("recall@10", 0.0),
            "best_val_recall@50": best_epoch_log.get("recall@50", 0.0),
            "best_val_recall@100": best_epoch_log.get("recall@100", 0.0),
            "best_val_arp": best_epoch_log.get("arp", 0.0),
        }
        results.append(res)

        print(
            f"Best:\n"
            f"R@10: {res['best_val_recall@10']:.6f} | "
            f"R@50: {res['best_val_recall@50']:.6f} | "
            f"R@100: {res['best_val_recall@100']:.6f} | "
            f"ARP: {res['best_val_arp']:.2f}"
        )

        model.save_model(configs.CHECKPOINT_DIR / f"implicit_mf_gamma_{config['gamma']}.npz")

    with open(configs.LOG_DIR / "gamma_search_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    if results:
        fieldnames = list(results[0].keys())
        with open(configs.LOG_DIR / "gamma_search_results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print("===============================================================")
    print("Gamma Comparison Complete!")
    for r in results:
        print(
            f"gamma={r['gamma']:.2f} -> "
            f"R@10: {r['best_val_recall@10']:.6f} | ARP: {r['best_val_arp']:.2f}"
        )
    print("===============================================================")

def main():
    parser = argparse.ArgumentParser(description="ImplicitMF train API / CLI")
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "grid", "gamma"],
        help="실행 모드 선택 (single: 단일 파라미터 학습, grid: 베이스라인 탐색, gamma: 편향 보정 비교)",
    )
    parser.add_argument("--k_dim", type=int, default=64, help="잠재 벡터 차원 수")
    parser.add_argument("--alpha", type=float, default=40.0, help="신뢰도 스케일링 계수")
    parser.add_argument("--gamma", type=float, default=0.0, help="인기도 편향 보정 강도")
    parser.add_argument("--reg_weight", type=float, default=0.01, help="L2 정규화 계수")
    parser.add_argument("--iteration", type=int, default=20, help="최대 학습 에포크 수")
    parser.add_argument("--patience", type=int, default=5, help="조기 종료 patience")
    parser.add_argument("--random_state", type=int, default=777, help="난수 시드")
    parser.add_argument("--top_k", type=int, default=10, help="기본 Top-K 개수")
    parser.add_argument("--save_path", type=str, default=None, help="single 모드 모델 저장 경로")
    
    args = parser.parse_args()
    
    if args.mode == "single":
        train_single(
            k_dim=args.k_dim,
            alpha=args.alpha,
            gamma=args.gamma,
            reg_weight=args.reg_weight,
            iteration=args.iteration,
            patience=args.patience,
            random_state=args.random_state,
            top_k=args.top_k,
            save_path=args.save_path,
        )
    elif args.mode == "grid":
        grid_serach()
    elif args.mode == "gamma":
        compare_with_diff_gamma()

if __name__ == "__main__":
    main()