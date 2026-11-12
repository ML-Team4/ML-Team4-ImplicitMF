import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from implicit_mf import ImplicitMF
from utils.utils import parse_target_dict, prepare_for_test, plot_gamma_tradeoff
import gc
import csv
import json


CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
GAMMA_LIST = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

def test():
    counts, items, test_data = prepare_for_test()
    test_users, test_targets = test_data
    test_target_dict = parse_target_dict(test_targets)
    train_listener_counts = items["train_listener_count"].to_numpy()

    results = []
    
    for gamma in GAMMA_LIST:
        weight_path = CHECKPOINT_DIR / f"implicit_mf_gamma_{gamma}.npz"

        model = ImplicitMF()
        model.load_model(weight_path)

        print(
            f"Loaded ({weight_path.name}): "
            f"k_dim={model.k_dim} | alpha={model.alpha} | "
            f"gamma={model.gamma} | reg_weight={model.reg_weight}"
        )

        metrics = model.evaluate(
            eligible_users=test_users,
            target_dict=test_target_dict,
            train_counts=counts,
            train_listener_counts=train_listener_counts,
            top_ks=[10, 50, 100],
        )

        res = {
            "k_dim": model.k_dim,
            "alpha": model.alpha,
            "gamma": model.gamma,
            "reg_weight": model.reg_weight,
            "test_recall@10": metrics["recall@10"],
            "test_recall@50": metrics["recall@50"],
            "test_recall@100": metrics["recall@100"],
            "test_arp": metrics["arp"],
        }
        results.append(res)

        print(
            f" -> [TEST] gamma={model.gamma:.2f} | "
            f"R@10: {res['test_recall@10']:.6f} | "
            f"R@50: {res['test_recall@50']:.6f} | "
            f"R@100: {res['test_recall@100']:.6f} | "
            f"ARP: {res['test_arp']:.2f}\n"
        )

        # 다음 모델 로드 전 이전 잠재 행렬(k_dim=256) 메모리 해제
        del model
        gc.collect()

    print(f"=======================================================================================")
    print(
        f"{'Gamma':>6} | {'Test R@10':>12} | {'Test R@50':>12} | "
        f"{'Test R@100':>12} | {'Test ARP':>10}"
    )
    print(f"=======================================================================================")
    for r in results:
        print(
            f"{r['gamma']:>6.2f} | {r['test_recall@10']:>12.6f} | "
            f"{r['test_recall@50']:>12.6f} | {r['test_recall@100']:>12.6f} | "
            f"{r['test_arp']:>10.2f}"
        )
    print(f"=======================================================================================")

    with open(LOG_DIR / "test_gamma_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    if results:
        with open(
            LOG_DIR / "test_gamma_results.csv", "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        plot_gamma_tradeoff(
            results=results,
            save_path=LOG_DIR / "test_gamma_tradeoff.png"
        )

if __name__ == "__main__":
    test()