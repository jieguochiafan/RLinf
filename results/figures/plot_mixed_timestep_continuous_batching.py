"""Draw a Gantt-style diagram for mixed-timestep continuous batching."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main() -> None:
    requests = [
        ("Req A", 0),
        ("Req B", 1),
        ("Req C", 2),
        ("Req D", 3),
        ("Req E", 4),
    ]
    denoise_steps = 5
    max_kernel = max(arrival + denoise_steps for _, arrival in requests)

    colors = {
        0: "#4C78A8",
        1: "#72B7B2",
        2: "#F58518",
        3: "#E45756",
        4: "#54A24B",
    }

    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    row_height = 0.72

    for row_idx, (request_name, arrival_kernel) in enumerate(requests):
        y = len(requests) - row_idx - 1
        for step in range(denoise_steps):
            kernel = arrival_kernel + step
            rect = Rectangle(
                (kernel + 0.08, y - row_height / 2),
                0.84,
                row_height,
                facecolor=colors[step],
                edgecolor="white",
                linewidth=1.4,
            )
            ax.add_patch(rect)
            ax.text(
                kernel + 0.5,
                y,
                f"{request_name[-1]}{step}",
                ha="center",
                va="center",
                color="white",
                fontsize=11,
                fontweight="bold",
            )
        ax.text(
            arrival_kernel - 0.08,
            y - 0.43,
            "arrive",
            ha="right",
            va="center",
            fontsize=8.5,
            color="#555555",
        )

    for kernel in range(max_kernel):
        ax.axvline(kernel, color="#D9D9D9", linewidth=0.8, zorder=0)
        active = []
        for request_name, arrival_kernel in requests:
            step = kernel - arrival_kernel
            if 0 <= step < denoise_steps:
                active.append(f"{request_name[-1]}{step}")
        if active:
            ax.text(
                kernel + 0.5,
                -0.88,
                "[" + ",".join(active) + "]",
                ha="center",
                va="top",
                fontsize=9,
                color="#333333",
                rotation=18,
            )

    ax.annotate(
        "New requests join the next AE launch",
        xy=(4.5, len(requests) - 1.2),
        xytext=(5.8, len(requests) + 0.05),
        arrowprops={"arrowstyle": "->", "linewidth": 1.0, "color": "#333333"},
        fontsize=10,
        color="#222222",
    )
    ax.annotate(
        "Completed requests leave the batch",
        xy=(5.0, len(requests) - 1.0),
        xytext=(6.2, len(requests) - 0.65),
        arrowprops={"arrowstyle": "->", "linewidth": 1.0, "color": "#333333"},
        fontsize=10,
        color="#222222",
    )

    ax.set_xlim(0, max_kernel)
    ax.set_ylim(-1.7, len(requests) + 0.55)
    ax.set_xticks([idx + 0.5 for idx in range(max_kernel)])
    ax.set_xticklabels([f"K{idx}" for idx in range(max_kernel)], fontsize=10)
    ax.set_yticks(range(len(requests)))
    ax.set_yticklabels([name for name, _ in reversed(requests)], fontsize=11)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("AE batch", labelpad=12, fontsize=11)
    ax.xaxis.set_label_position("top")
    ax.set_title(
        "Mixed-Timestep Continuous Batching for the Action Expert",
        fontsize=14,
        fontweight="bold",
        pad=34,
    )
    ax.text(
        -0.05,
        -0.88,
        "GPU batch",
        ha="right",
        va="top",
        fontsize=10,
        color="#222222",
        fontweight="bold",
    )

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("#FBFBFB")
    fig.patch.set_facecolor("white")

    output_dir = Path(__file__).resolve().parent
    fig.tight_layout()
    fig.savefig(output_dir / "mixed_timestep_continuous_batching_gantt.png", dpi=240)
    fig.savefig(output_dir / "mixed_timestep_continuous_batching_gantt.pdf")


if __name__ == "__main__":
    main()
