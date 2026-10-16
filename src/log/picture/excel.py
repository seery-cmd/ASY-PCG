import pandas as pd


# 输入文件
input_file = "C:/Users/lenovo/CLionProjects/neon_spmv/src/log/result/F32_asy.txt"

# 输出 Excel 文件
output_file = "FP32_asy.xlsx"

# 定义列名
columns = [
    "matrix",
    "rows",
    "cols",
    "nnz",
    "time_csr_cpu",
    "time_csr_neon",
    "time_ell_neon",
    "time_ascr_neon",
    "time_sy_neon",
    "time_kml",
    "nnz_csr",
    "nnz_ell",
    "nnz_acsr",
    "nnz_sy",]


# 读取数据
df = pd.read_csv(
    input_file,
    sep=r"\s+",
    header=None,
    names=columns
)

# 写入 Excel
df.to_excel(
    output_file,
    index=False,
    engine="openpyxl"
)