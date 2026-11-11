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


def grid_serach():
    datasets = load_data()
    counts = datasets["counts"]
    items = datasets["items"]
    val_data = datasets["evaluation"]["validation"]

    param_grid = {
        "k_dim": [64, 128, 256],         
        "alpha": [10.0, 20.0, 40.0],     
        "gamma": [0.0],   
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

        # 역대 최고 Recall@10 달성 시 체크포인트 저장
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
    print(f" Best R@10: {best_recall_10:.6f}")
    print(f"Best Config: {best_config}")
    print("=====================")

if __name__ == "__main__":
    grid_serach()

