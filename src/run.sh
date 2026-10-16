#!/bin/bash
target_dir="/home/data/suitesparse_collection"
start_name="windtunnel_evap2d"     # 从这个名字开始输出
started=0           # 开关，0=还没到，1=已经开始

# 开启 nullglob，避免目录为空时 for 循环直接使用字面量 *
shopt -s nullglob

# 使用 for 循环遍历目录下的所有条目
for item in "$target_dir"/*; do
    # 判断是否为目录（文件夹）
    if [ -d "$item" ]; then
        # 打印文件夹名称（不含路径）
        name=$(basename "$item")
        if [ "$name" = "$start_name" ]; then
            started=1
        fi
        if [ "$started" -eq 1 ]; then
        make run MTX=$target_dir"/$name""/$name.mtx" LOG=/home/hw_EMatrix/seery/src/log/result/FP32_asy.log
        fi
    fi
done