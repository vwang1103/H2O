import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Data
# -----------------------------
df = pd.DataFrame({
    "Workload": ["Small", "Small", "Large", "Large", "Stress", "Stress"],
    "Method": ["Full Cache", "H2O", "Full Cache", "H2O", "Full Cache", "H2O"],
    "Batch": [4, 4, 16, 16, 24, 24],
    "Prompt/Gen": ["512/512", "512/512", "2048/2048", "2048/2048", "2048/2048", "2048/2048"],
    "Peak_GPU_Mem_GB": [15.132, 13.542, 62.236, 36.527, np.nan, 48.391],
    "Throughput_tok_s": [273.293, 203.767, 531.737, 720.116, np.nan, 938.937],
    "Latency_s": [7.494, 10.051, 61.624, 45.504, np.nan, 52.349],
})

workloads = ["Small", "Large", "Stress"]
methods = ["Full Cache", "H2O"]

# -----------------------------
# Helper: grouped bar chart
# -----------------------------
def grouped_bar(metric, ylabel, title, filename):
    x = np.arange(len(workloads))
    width = 0.35

    plt.figure(figsize=(8, 4.8))

    for i, method in enumerate(methods):
        values = [
            df[(df["Workload"] == w) & (df["Method"] == method)][metric].values[0]
            for w in workloads
        ]

        bars = plt.bar(
            x + (i - 0.5) * width,
            [0 if np.isnan(v) else v for v in values],
            width,
            label=method
        )

        for bar, v in zip(bars, values):
            if np.isnan(v):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    1,
                    "OOM",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold"
                )
            else:
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{v:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9
                )

    plt.xticks(x, workloads)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


# -----------------------------
# Plot 1: Peak GPU Memory
# -----------------------------
grouped_bar(
    "Peak_GPU_Mem_GB",
    "Peak GPU Memory (GB)",
    "Peak GPU Memory Comparison",
    "gpu_memory_comparison.png"
)

# -----------------------------
# Plot 2: Throughput
# -----------------------------
grouped_bar(
    "Throughput_tok_s",
    "Throughput (tokens/sec)",
    "Throughput Comparison",
    "throughput_comparison.png"
)

# -----------------------------
# Plot 3: Latency
# -----------------------------
grouped_bar(
    "Latency_s",
    "Latency (seconds)",
    "Latency Comparison",
    "latency_comparison.png"
)

# -----------------------------
# Plot 4: Efficiency Scatter Plot with Reference Lines
# -----------------------------
plt.figure(figsize=(7.5, 5.5))

valid = df.dropna(subset=["Peak_GPU_Mem_GB", "Throughput_tok_s"])

for _, row in valid.iterrows():
    plt.scatter(row["Peak_GPU_Mem_GB"], row["Throughput_tok_s"], s=90)
    plt.text(
        row["Peak_GPU_Mem_GB"] + 0.8,
        row["Throughput_tok_s"],
        f'{row["Workload"]} {row["Method"]}',
        fontsize=9
    )

# Reference efficiency lines: throughput = efficiency * memory
x_line = np.linspace(0, valid["Peak_GPU_Mem_GB"].max() * 1.1, 100)

for eff in [5, 10, 15, 20]:
    y_line = eff * x_line
    plt.plot(x_line, y_line, linestyle="--", linewidth=1)
    plt.text(
        x_line[-1],
        y_line[-1],
        f"{eff} tok/s/GB",
        fontsize=8,
        va="center"
    )

plt.xlabel("Peak GPU Memory (GB)")
plt.ylabel("Throughput (tokens/sec)")
plt.title("Memory–Throughput Efficiency")
plt.xlim(0, valid["Peak_GPU_Mem_GB"].max() * 1.2)
plt.ylim(0, valid["Throughput_tok_s"].max() * 1.15)
plt.tight_layout()
plt.savefig("memory_throughput_efficiency_with_reference_lines.png", dpi=300)
plt.show()
