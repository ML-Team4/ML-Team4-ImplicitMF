from implicit_mf import ImplicitMF
from load_datasets import load_data
import csv
import json
from itertools import product
from pathlib import Path

CHECKPOINT_DIR = Path("checkpoints")
LOG_DIR = Path("logs")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_WEIGHT_DIR = Path("./checkpoints/grid_best_implicit_mf.npz")

def prepare_for_train():
    datasets = load_data()
    counts = datasets["counts"]
    items = datasets["items"]
    val_data = datasets["evaluation"]["validation"]

    return counts, items, val_data

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
            model.save_model(CHECKPOINT_DIR / "grid_best_implicit_mf.npz")

    with open(LOG_DIR / "grid_search_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    if results:
        fieldnames = list(results[0].keys())
        with open(LOG_DIR / "grid_search_results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print("=====================")
    print(f"Best R@10: {best_recall_10:.6f}")
    print(f"Best Config: {best_config}")
    print("=====================")

def compare_with_diff_gamma():
    """
    Grid Search로 찾은 최적 하이퍼파라미터를 고정한 상태에서
    gamma 값만 변경하며 validation set 기준으로 편향 완화(ARP) 및 성능(Recall)을 비교
    """
    counts, items, val_data = prepare_for_train()

    base_model = ImplicitMF()
    base_model.load_model(MODEL_WEIGHT_DIR)

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

        model.save_model(CHECKPOINT_DIR / f"implicit_mf_gamma_{config['gamma']}.npz")

    with open(LOG_DIR / "gamma_search_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    if results:
        fieldnames = list(results[0].keys())
        with open(LOG_DIR / "gamma_search_results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print("=====================")
    print("Gamma Comparison Complete!")
    for r in results:
        print(
            f"gamma={r['gamma']:.2f} -> "
            f"R@10: {r['best_val_recall@10']:.6f} | ARP: {r['best_val_arp']:.2f}"
        )
    print("=====================")

if __name__ == "__main__":
    #grid_serach()
    compare_with_diff_gamma()
