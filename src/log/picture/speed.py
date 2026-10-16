from openpyxl import load_workbook
import matplotlib.pyplot as plt
import numpy as np
# Excel 文件路径
file_path = "FP32_asy.xlsx"
datasets = []
# 打开 Excel
workbook = load_workbook(file_path, data_only=True)

# 选择工作表
worksheet = workbook["Sheet1"]

# 读取 S2 单元格
for i in range(2,939):
    row = [worksheet[f"{col}{i}"].value for col in ["S", "T", "U", "W", "V"]]
    datasets.append(row)

data = np.array(datasets, dtype=float)   # shape = (444, 5)

# 每列对应一个箱型图
box_data = [data[:, j] for j in range(data.shape[1])]
fig, ax = plt.subplots(figsize=(8, 6))
# 绘制箱型图
bp = ax.boxplot(
    box_data,
    widths=0.3,
    patch_artist=True,
    medianprops=dict(color="black", linewidth=2),
    whiskerprops=dict(linestyle='-'),
    flierprops=dict(marker="o", markerfacecolor="gray",
                    markeredgecolor="gray", markersize=4))
# 箱体颜色
def rgb(r, g, b):
    return (r / 255, g / 255, b / 255)

colors = [
    rgb(187, 205, 218),   # 第 1 个箱子
    rgb(240, 168, 109),   # 第 2 个箱子
    rgb( 88, 160, 210),   # 第 3 个箱子
    '#D8352D',
    rgb( 12, 199, 182),   # 第 4 个箱子
]
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(colors[i])

ax.set_ylim(0.0, 3.0)
# ---------- 去掉 X 轴刻度 ----------
ax.set_xticks([])
ax.tick_params(axis='x', which='both', length=0)

# ---------- 放大 Y 轴刻度 ----------
ax.tick_params(axis='y', labelsize=20)

ax.set_ylabel("Speed Up", fontsize=22)

ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("FP32_asy.png", dpi=600,bbox_inches='tight')
plt.show()