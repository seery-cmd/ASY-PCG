from openpyxl import load_workbook
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter, LogLocator, NullFormatter
# Excel 文件路径
file_path = "petsc-double.xlsx"
datasets = []
# 打开 Excel
workbook = load_workbook(file_path, data_only=True)

# 选择工作表
worksheet = workbook["Sheet1"]

# 读取 S2 单元格
for i in range(2,116):
    row = [worksheet[f"{col}{i}"].value for col in ["L", "M", "N","O"]]
    datasets.append(row)

data = np.array(datasets, dtype=float)
# ---------- 2. 绘图参数 ----------
methods = ["NONE", "Jacobi", "IC", "SSOR"]
markers = ["o", "s", "^", "D"]
colors  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# 左右宽度比 8:3，高度 6
fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(8 + 3, 6),
    gridspec_kw={"width_ratios": [8, 3]}
)

# ========== 左侧：散点图（对数 y 轴） ==========
x = np.arange(data.shape[0])
for j in range(4):
    ax1.scatter(x, data[:, j],
                marker=markers[j],
                color=colors[j],
                s=22,
                alpha=0.75,
                edgecolors="none",
                label=methods[j])

ax1.set_yscale("log")
ax1.set_ylim(1e-1, 1e3)                     # 10^-1 ~ 10^3
ax1.yaxis.set_major_locator(LogLocator(base=10.0))
ax1.yaxis.set_minor_formatter(NullFormatter())

# 只保留主刻度（10^-1, 10^0, 10^1, 10^2, 10^3）的虚线
ax1.grid(True, which="major", linestyle="--", alpha=0.35)
ax1.axhline(y=1, color="red", linestyle="--", linewidth=1.2, zorder=3)
ax1.set_xlabel("Workloads", fontsize=18)
ax1.set_ylabel("Speed Up", fontsize=18)
ax1.legend(fontsize=16, frameon=True, loc="upper left")
ax1.tick_params(axis="x", which="both",
                bottom=False, top=False,
                labelbottom=False)          # 取消 x 轴刻度
ax1.tick_params(axis="y", which="major", labelsize=14)
ax1.tick_params(axis="y", which="minor", labelsize=14)
# ========== 右侧：柱状图（保留外框，无刻度，柱顶显示百分比） ==========
values = [73.91, 73.10, 63.68, 71.91]

xpos = np.arange(len(values))
bars = ax2.bar(xpos, values,
               color=colors,
               edgecolor="black",
               linewidth=0.8,
               width=0.5)                   # 柱子宽度保持不变

# 去掉刻度和标签，保留外框
ax2.set_xticks([])
ax2.set_yticks([])
ax2.tick_params(axis="both", which="both",
                bottom=False, left=False,
                labelbottom=False, labelleft=False)

for spine in ax2.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.9)

# 让柱子显得短一点：抬高 y 轴上限，留出空间显示数值
ax2.set_ylim(0, max(values) * 1.25)

# 在每根柱子上方显示百分比
for bar, val in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             val + 1.5,                     # 略微高于柱顶
             f"{val:.2f}%",
             ha="center", va="bottom",
             rotation=90,
             fontsize=12, fontweight="bold")

# 控制两侧留白，让柱子填满绘图区宽度
ax2.set_xlim(-0.5, len(values) - 0.5)

# ---------- 3. 消除左右子图之间的空白 ----------
plt.subplots_adjust(wspace=0)               # 关键：水平间距设为 0

# 可选：手动微调边距，避免标签被裁剪
plt.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.95)

# ---------- 4. 保存 & 显示 ----------
plt.savefig("petsc.pdf", dpi=300, bbox_inches="tight")
plt.show()