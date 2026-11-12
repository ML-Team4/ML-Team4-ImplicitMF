from Scripts.load_datasets import load_data
import pyarrow as pa
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

def prepare_for_train():
    datasets = load_data()
    counts = datasets["counts"]
    items = datasets["items"]
    val_data = datasets["evaluation"]["validation"]

    return counts, items, val_data

def prepare_for_test():
    datasets = load_data()
    counts = datasets["counts"]
    items = datasets["items"]
    test_data = datasets["evaluation"]["test"]

    return counts, items, test_data

def build_target_dict(targets_table) -> dict[int, set[int]]:
    """_summary_
        PyArrow Table 형태를 {user_idx: set(item_idx)} dict 로 변환
    Args:
        targets_table (_type_): _description_

    Returns:
        dict[int, set[int]]: _description_
    """
    target_dict: dict[int, set[int]] = {}
    for u, i in zip(
        targets_table["user_idx"].to_numpy(),
        targets_table["item_idx"].to_numpy(),
    ):
        target_dict.setdefault(int(u), set()).add(int(i))
    return target_dict

def parse_target_dict(
        targets: pa.Table | dict[int, set[int]]
    ) -> dict[int, set[int]]:
        """_summary_
            targets PyArrow -> {user_idx: set(item_idx)} dict 으로 변환
        Args:
            targets (pa.Table | dict[int, set[int]]): PyArrow Table or dict

        Returns:
            dict[int, set[int]]: {user_idx: set(item_idx)} dict
        """
        if isinstance(targets, dict):
            return targets

        target_u = targets["user_idx"].to_numpy().tolist()
        target_i = targets["item_idx"].to_numpy().tolist()

        target_dict = {}
        for u, i in zip(target_u, target_i):
            if u not in target_dict:
                target_dict[u] = set()
            target_dict[u].add(i)
        return target_dict

def parse_listener_counts(
        items_dataset: pa.Table | np.ndarray | None
    ) -> np.ndarray | None:
        """_summary_
            items.csv <- train_listener_count 추출 및 np.ndarray 변환
        Args:
            items_dataset (pa.Table | np.ndarray | None): items.csv

        Returns:
            np.ndarray | None: train_listener_count 1D array
        """
        if items_dataset is None:
            return None
        if isinstance(items_dataset, np.ndarray):
            return items_dataset
        if (
            hasattr(items_dataset, "column_names")
            and "train_listener_count" in items_dataset.column_names
        ):
            return items_dataset["train_listener_count"].to_numpy()
        return np.asarray(items_dataset)

def plot_gamma_tradeoff(results: list[dict], save_path: Path) -> None:
    gammas = [r["gamma"] for r in results]
    recalls = [r["test_recall@10"] for r in results]
    arps = [r["test_arp"] for r in results]

    fig, ax1 = plt.subplots(figsize=(9, 5.5), dpi=300)

    # 왼쪽 Y축 Recall@10
    color_recall = "#1f77b4"
    line1 = ax1.plot(
        gammas,
        recalls,
        color=color_recall,
        marker="o",
        linewidth=2.2,
        markersize=7,
        label="Recall@10 (Accuracy)",
    )
    ax1.set_xlabel("IPS Propensity Exponent ($\\gamma$)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Test Recall@10", color=color_recall, fontsize=12, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_recall)
    ax1.set_xticks(gammas)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Y축 여백 확보
    r_margin = (max(recalls) - min(recalls)) * 0.25 if len(recalls) > 1 else 0.001
    ax1.set_ylim(min(recalls) - r_margin, max(recalls) + r_margin)

    for g, r in zip(gammas, recalls):
        ax1.annotate(
            f"{r:.5f}",
            (g, r),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8.5,
            color=color_recall,
            fontweight="bold",
        )

    # 오른쪽 Y축 ARP 
    ax2 = ax1.twinx()
    color_arp = "#d62728"
    line2 = ax2.plot(
        gammas,
        arps,
        color=color_arp,
        marker="s",
        linestyle="--",
        linewidth=2.2,
        markersize=7,
        label="ARP (Popularity Bias)",
    )
    ax2.set_ylabel(
        "Test ARP (Avg. Listener Count)", color=color_arp, fontsize=12, fontweight="bold"
    )
    ax2.tick_params(axis="y", labelcolor=color_arp)

    a_margin = (max(arps) - min(arps)) * 0.25 if len(arps) > 1 else 5.0
    ax2.set_ylim(min(arps) - a_margin, max(arps) + a_margin)

    for g, a in zip(gammas, arps):
        ax2.annotate(
            f"{a:.2f}",
            (g, a),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            fontsize=8.5,
            color=color_arp,
            fontweight="bold",
        )

    lines = line1 + line2 
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", frameon=True, fontsize=9.5)

    plt.title(
        "Trade-off between Recall@10 and ARP across $\\gamma$",
        fontsize=12.5,
        fontweight="bold",
        pad=14,
    )
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved trade-off graph to: {save_path}")

def extract_item_meta(items: pa.Table, item_indices: np.ndarray) -> list[dict]:
    col_names = set(items.column_names)
    title_col = next(
        (c for c in ["recording_name", "track_name", "title", "name"] if c in col_names),
        None,
    )
    artist_col = next(
        (c for c in ["artist_name", "artist", "creator"] if c in col_names),
        None,
    )
    pop_col = "train_listener_count" if "train_listener_count" in col_names else None

    rows = items.take(item_indices.tolist()).to_pylist()
    meta_list = []
    for idx, row in zip(item_indices.tolist(), rows):
        meta_list.append(
            {
                "item_idx": int(idx),
                "title": str(row[title_col]) if title_col else f"item_{idx}",
                "artist": str(row[artist_col]) if artist_col else "-",
                "listeners": int(row[pop_col]) if pop_col else 0,
            }
        )
    return meta_list

def export_markdown_report(
    md_path: Path,
    user_indices: list[int],
    histories: dict[int, tuple[int, list[dict]]],
    recs_by_gamma: dict[float, dict[int, list[dict]]],
    top_k: int,
) -> None:
    def esc(text: str) -> str:
        return text.replace("|", "\\|").replace("\n", " ")

    lines = ["# ImplicitMF 정성적 추천 결과 리포트\n"]

    for u in user_indices:
        total_tracks, hist_metas = histories[u]
        lines.append(f"## User `{u}` (학습 기간 총 {total_tracks:,}곡 청취)\n")

        lines.append("### 학습 기간 주요 청취 이력 (Top-5)\n")
        lines.append("| Rank | Track Title | Artist | Play Count | Train Listeners |")
        lines.append("| :---: | :--- | :--- | :---: | :---: |")
        for rank, m in enumerate(hist_metas, 1):
            lines.append(
                f"| {rank} | {esc(m['title'])} | {esc(m['artist'])} | "
                f"{m['play_count']}회 | {m['listeners']:,}명 |"
            )
        lines.append("")

        for gamma, user_map in recs_by_gamma.items():
            metas = user_map[u]
            avg_pop = float(np.mean([m["listeners"] for m in metas])) if metas else 0.0
            tag = "Baseline (IPS 미적용)" if gamma == 0.0 else f"IPS 적용 ($\\gamma={gamma:.2f}$)"

            lines.append(f"### Top-{top_k} 추천 목록 — {tag} (평균 청취자 수: **{avg_pop:.1f}명**)\n")
            lines.append("| Rank | Item ID | Track Title | Artist | Train Listeners |")
            lines.append("| :---: | :---: | :--- | :--- | :---: |")
            for rank, m in enumerate(metas, 1):
                lines.append(
                    f"| {rank} | `{m['item_idx']}` | {esc(m['title'])} | "
                    f"{esc(m['artist'])} | {m['listeners']:,}명 |"
                )
            lines.append("")

        lines.append("---\n")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"마크다운 리포트 저장 완료: {md_path}")

def setup_matplotlib_font() -> None:
    plt.rcParams["axes.unicode_minus"] = False
    preferred_fonts = ["NanumGothic", "Malgun Gothic", "AppleGothic", "Noto Sans CJK KR", "DejaVu Sans"]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in preferred_fonts:
        if font in available:
            plt.rcParams["font.family"] = font
            break

def plot_recommendation_comparison(
    save_dir: Path,
    user_indices: list[int],
    recs_by_gamma: dict[float, dict[int, list[dict]]],
    top_k: int,
) -> None:
    setup_matplotlib_font()
    gammas = list(recs_by_gamma.keys())
    n_models = len(gammas)

    for u in user_indices:
        fig, axes = plt.subplots(1, n_models, figsize=(6.8 * n_models, max(4.8, top_k * 0.48)), dpi=300)
        if n_models == 1:
            axes = [axes]

        # 두 모델의 X축 스케일을 동일하게 맞춰 인기도 차이를 직관적으로 비교
        max_pop = max(
            (m["listeners"] for g in gammas for m in recs_by_gamma[g][u]),
            default=10,
        )

        for ax, g in zip(axes, gammas):
            metas = recs_by_gamma[g][u]
            labels = [
                f"#{r} {m['title'][:22]} - {m['artist'][:14]}"
                for r, m in enumerate(metas, 1)
            ]
            pops = [m["listeners"] for m in metas]
            avg_pop = float(np.mean(pops)) if pops else 0.0

            color = "#4c72b0" if g == 0.0 else "#dd8452"
            y_pos = np.arange(len(metas))
            bars = ax.barh(y_pos, pops, color=color, alpha=0.88, height=0.65)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=8.5)
            ax.invert_yaxis()
            ax.set_xlim(0, max_pop * 1.22)
            ax.set_xlabel("Train Listener Count (Popularity)", fontsize=9.5, fontweight="bold")
            ax.grid(axis="x", linestyle="--", alpha=0.4)

            title_prefix = "Baseline (gamma=0.00)" if g == 0.0 else f"IPS Applied (gamma={g:.2f})"
            ax.set_title(
                f"{title_prefix}\nMean Listeners (ARP): {avg_pop:.1f}",
                fontsize=10.5,
                fontweight="bold",
                pad=10,
            )

            for bar, pop in zip(bars, pops):
                ax.text(
                    pop + max_pop * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{pop:,}",
                    va="center",
                    ha="left",
                    fontsize=8,
                    fontweight="bold",
                )

        fig.suptitle(
            f"User {u} Top-{top_k} Recommendation Popularity Comparison",
            fontsize=12.5,
            fontweight="bold",
            y=1.02,
        )
        fig.tight_layout()
        out_img = save_dir / f"user_{u}_rec_comparison.png"
        plt.savefig(out_img, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"추천 비교 차트 저장 완료: {out_img}")