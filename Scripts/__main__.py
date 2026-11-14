import os
import sys
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.utils import prepare_for_test, extract_item_meta, export_markdown_report, plot_recommendation_comparison
from Scripts.implicit_mf import ImplicitMF
import argparse
import csv
import gc
import numpy as np
import pyarrow as pa
import configs

CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

def get_user_history(user_idx: int, counts, items: pa.Table, top_n: int = 5) -> tuple[int, list[dict]]:
    row = counts.getrow(user_idx)
    if row.nnz == 0:
        return 0, []

    top_pos = np.argsort(-row.data)[:top_n]
    hist_items = row.indices[top_pos]
    hist_plays = row.data[top_pos]
    metas = extract_item_meta(items, hist_items)

    for m, play_cnt in zip(metas, hist_plays):
        m["play_count"] = int(play_cnt)
    return int(row.nnz), metas

def print_user_history(user_idx: int, total_tracks: int, hist_metas: list[dict]) -> None:
    if not hist_metas:
        print(f"[User {user_idx}] 학습 청취 이력이 없습니다.")
        return

    print(f"\n[User {user_idx}] 학습 기간 주요 청취 곡 (총 {total_tracks}곡 청취)")
    print("-" * 84)
    for rank, m in enumerate(hist_metas, 1):
        print(
            f"  {rank}. {m['title'][:34]:<34} | {m['artist'][:22]:<22} | "
            f"청취 {m['play_count']}회 (전체청취자 {m['listeners']:,}명)"
        )
    print("-" * 84)

def recommend_for_gamma(
    gamma: float,
    user_indices: list[int],
    counts,
    items: pa.Table,
    top_k: int,
) -> tuple[dict[int, list[dict]], list[dict]]:
    weight_path = configs.CHECKPOINT_DIR / f"implicit_mf_gamma_{gamma}.npz"

    model = ImplicitMF()
    model.load_model(weight_path)

    user_recs_map: dict[int, list[dict]] = {}
    flat_rows: list[dict] = []

    for u in user_indices:
        seen_items = counts.getrow(u).indices
        rec_indices = model.recommend(user_idx=u, train_seen=seen_items, top_k=top_k)
        metas = extract_item_meta(items, rec_indices)
        user_recs_map[u] = metas

        avg_pop = float(np.mean([m["listeners"] for m in metas])) if metas else 0.0
        print(
            f"\n[User {u} | gamma={gamma:.2f}] Top-{top_k} 추천 결과 "
            f"(추천곡 평균 청취자 수: {avg_pop:.1f}명)"
        )
        print(f"{'Rank':>4} | {'Item ID':>8} | {'Track Title':<34} | {'Artist':<22} | {'Listeners':>10}")
        print("-" * 90)

        for rank, m in enumerate(metas, 1):
            print(
                f"{rank:>4} | {m['item_idx']:>8} | {m['title'][:34]:<34} | "
                f"{m['artist'][:22]:<22} | {m['listeners']:>10,}"
            )
            flat_rows.append(
                {
                    "user_idx": u,
                    "gamma": gamma,
                    "rank": rank,
                    "item_idx": m["item_idx"],
                    "title": m["title"],
                    "artist": m["artist"],
                    "train_listener_count": m["listeners"],
                }
            )
        print("-" * 90)

    del model
    gc.collect()
    return user_recs_map, flat_rows


def main():
    parser = argparse.ArgumentParser(description="ImplicitMF (IPS 편향 보정 적용) 음악 추천 CLI")
    parser.add_argument("--user_idx", type=int, nargs="+", default=[0], help="추천 대상 유저 인덱스")
    parser.add_argument("--gamma", type=float, default=0.2, help="IPS 편향 보정 강도")
    parser.add_argument("--compare", action="store_true", help="Baseline(0.0)과 지정 gamma 비교")
    parser.add_argument("--top_k", type=int, default=10, help="추천 곡 수")
    args = parser.parse_args()

    counts, items, _ = prepare_for_test()
    num_users = counts.shape[0]

    histories: dict[int, tuple[int, list[dict]]] = {}
    for u in args.user_idx:
        if u < 0 or u >= num_users:
            raise ValueError(f"invalide user_idx={u}. (user_idx: 0 ~ {num_users - 1})")
        total_tracks, hist_metas = get_user_history(u, counts, items, top_n=5)
        histories[u] = (total_tracks, hist_metas)
        print_user_history(u, total_tracks, hist_metas)

    target_gammas = [0.0, args.gamma] if (args.compare and args.gamma != 0.0) else [args.gamma]

    recs_by_gamma: dict[float, dict[int, list[dict]]] = {}
    exported_rows: list[dict] = []

    for g in target_gammas:
        user_map, flat_rows = recommend_for_gamma(
            gamma=g,
            user_indices=args.user_idx,
            counts=counts,
            items=items,
            top_k=args.top_k,
        )
        recs_by_gamma[g] = user_map
        exported_rows.extend(flat_rows)

    if exported_rows:
        csv_path = configs.RESULT_DIR/ "recommendations.csv"
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(exported_rows[0].keys()))
            writer.writeheader()
            writer.writerows(exported_rows)
        print(f"\nsuccess to save: {csv_path}")

        export_markdown_report(
            md_path=configs.RESULT_DIR / "recommendations.md",
            user_indices=args.user_idx,
            histories=histories,
            recs_by_gamma=recs_by_gamma,
            top_k=args.top_k,
        )

        plot_recommendation_comparison(
            save_dir=configs.RESULT_DIR,
            user_indices=args.user_idx,
            recs_by_gamma=recs_by_gamma,
            top_k=args.top_k,
        )
if __name__ == "__main__":
    main()
