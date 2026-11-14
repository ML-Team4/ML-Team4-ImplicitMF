import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from implicit_mf import ImplicitMF
from utils.utils import parse_target_dict, prepare_for_test, plot_gamma_tradeoff
import gc
import csv
import json
import argparse

import configs

def evaluate_checkpoint(
    weight_path: Path,
    test_users,
    test_target_dict: dict[int, set[int]],
    counts,
    train_listener_counts,
    top_ks: list[int],
) -> dict:
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
        top_ks=top_ks,
    )

    res = {
        "k_dim": model.k_dim,
        "alpha": model.alpha,
        "gamma": model.gamma,
        "reg_weight": model.reg_weight,
        "test_recall@10": metrics.get("recall@10", 0.0),
        "test_recall@50": metrics.get("recall@50", 0.0),
        "test_recall@100": metrics.get("recall@100", 0.0),
        "test_arp": metrics.get("arp", 0.0),
    }

    del model
    gc.collect()
    return res

def evaluate_single(
    gamma: float = 0.2,
    weight_path: str | Path | None = None,
    top_ks: list[int] | None = None,
) -> dict:
    top_ks = top_ks or [10,]
    counts, items, test_data = prepare_for_test()
    test_users, test_targets = test_data
    test_target_dict = parse_target_dict(test_targets)
    train_listener_counts = items["train_listener_count"].to_numpy()

    target_path = (
        Path(weight_path)
        if weight_path
        else configs.CHECKPOINT_DIR / f"implicit_mf_gamma_{gamma}.npz"
    )

    res = evaluate_checkpoint(
        weight_path=target_path,
        test_users=test_users,
        test_target_dict=test_target_dict,
        counts=counts,
        train_listener_counts=train_listener_counts,
        top_ks=top_ks,
    )

    print(
        f" -> [TEST] gamma={res['gamma']:.2f} | "
        f"R@10: {res['test_recall@10']:.6f} | "
        f"R@50: {res['test_recall@50']:.6f} | "
        f"R@100: {res['test_recall@100']:.6f} | "
        f"ARP: {res['test_arp']:.2f}"
    )
    return res

def evaluate_compare_with_gamma(
    baseline_gamma: float = 0.0,
    final_gamma: float = 0.2,
    top_ks: list[int] | None = None,
) -> list[dict]:
    top_ks = top_ks or [10,]
    counts, items, test_data = prepare_for_test()
    test_users, test_targets = test_data
    test_target_dict = parse_target_dict(test_targets)
    train_listener_counts = items["train_listener_count"].to_numpy()

    results = []
    for gamma in [baseline_gamma, final_gamma]:
        weight_path = configs.CHECKPOINT_DIR / f"implicit_mf_gamma_{gamma}.npz"
        res = evaluate_checkpoint(
            weight_path=weight_path,
            test_users=test_users,
            test_target_dict=test_target_dict,
            counts=counts,
            train_listener_counts=train_listener_counts,
            top_ks=top_ks,
        )
        results.append(res)

    base, final = results[0], results[1]
    r10_diff = ((final["test_recall@10"] - base["test_recall@10"]) / base["test_recall@10"]) * 100
    arp_diff = ((final["test_arp"] - base["test_arp"]) / base["test_arp"]) * 100

    print("=" * 80)
    print(f"{'Model':<22} | {'R@10':>10} | {'R@50':>10} | {'R@100':>10} | {'ARP':>10}")
    print("-" * 80)
    for label, r in [("Baseline (gamma=0.0)", base), (f"Final (gamma={final_gamma})", final)]:
        print(
            f"{label:<22} | {r['test_recall@10']:>10.6f} | "
            f"{r['test_recall@50']:>10.6f} | {r['test_recall@100']:>10.6f} | "
            f"{r['test_arp']:>10.2f}"
        )
    print("-" * 80)
    print(f"Relative Change vs Baseline -> Recall@10: {r10_diff:+.2f}% | ARP: {arp_diff:+.2f}%")
    print("=" * 80)

    with open(configs.LOG_DIR / "final_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    return results

def evaluate_gamma_sweep(
    gamma_list: list[float] | None = None,
    top_ks: list[int] | None = None,
    save_outputs: bool = True,
) -> list[dict]:
    gamma_list = gamma_list or configs.GAMMA_LIST
    top_ks = top_ks or [10,]

    counts, items, test_data = prepare_for_test()
    test_users, test_targets = test_data
    test_target_dict = parse_target_dict(test_targets)
    train_listener_counts = items["train_listener_count"].to_numpy()

    results = []
    for gamma in gamma_list:
        weight_path = configs.CHECKPOINT_DIR / f"implicit_mf_gamma_{gamma}.npz"
        res = evaluate_checkpoint(
            weight_path=weight_path,
            test_users=test_users,
            test_target_dict=test_target_dict,
            counts=counts,
            train_listener_counts=train_listener_counts,
            top_ks=top_ks,
        )
        results.append(res)

        print(
            f" -> [TEST] gamma={res['gamma']:.2f} | "
            f"R@10: {res['test_recall@10']:.6f} | "
            f"R@50: {res['test_recall@50']:.6f} | "
            f"R@100: {res['test_recall@100']:.6f} | "
            f"ARP: {res['test_arp']:.2f}\n"
        )

    print("=" * 87)
    print(
        f"{'Gamma':>6} | {'Test R@10':>12} | {'Test R@50':>12} | "
        f"{'Test R@100':>12} | {'Test ARP':>10}"
    )
    print("=" * 87)
    for r in results:
        print(
            f"{r['gamma']:>6.2f} | {r['test_recall@10']:>12.6f} | "
            f"{r['test_recall@50']:>12.6f} | {r['test_recall@100']:>12.6f} | "
            f"{r['test_arp']:>10.2f}"
        )
    print("=" * 87)

    if save_outputs and results:
        with open(configs.LOG_DIR / "test_gamma_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        with open(
            configs.LOG_DIR / "test_gamma_results.csv", "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

        plot_gamma_tradeoff(
            results=results,
            save_path=configs.RESULT_DIR / "test_gamma_tradeoff.png",
        )

    return results

def main():
    parser = argparse.ArgumentParser(description="ImplicitMF 모델 Test 평가 API / CLI")
    parser.add_argument(
        "--mode",
        type=str,
        default="compare",
        choices=["compare", "single", "sweep"],
        help="실행 모드 (compare: Baseline vs 최종 모델 1:1 비교, single: 단일 모델 평가, sweep: 전체 gamma 민감도 분석)",
    )
    parser.add_argument("--gamma", type=float, default=0.2, help="평가할 대상 gamma 값")
    parser.add_argument("--baseline_gamma", type=float, default=0.0, help="비교 기준 Baseline gamma 값 (compare)")
    parser.add_argument("--weight_path", type=str, default=None, help="직접 지정할 체크포인트(.npz) 경로 (single)")
    parser.add_argument(
        "--top_ks",
        type=int,
        nargs="+",
        default=[10,],
        help="평가할 Top-K 목록 (ex: --top_ks 10 50 100)",
    )

    args = parser.parse_args()

    if args.mode == "compare":
        evaluate_compare_with_gamma(
            baseline_gamma=args.baseline_gamma,
            final_gamma=args.gamma,
            top_ks=args.top_ks,
        )
    elif args.mode == "single":
        evaluate_single(
            gamma=args.gamma,
            weight_path=args.weight_path,
            top_ks=args.top_ks,
        )
    elif args.mode == "sweep":
        evaluate_gamma_sweep(top_ks=args.top_ks)

if __name__ == "__main__":
    main()
    
    
