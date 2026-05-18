#!/bin/bash

# ===================== 配置项 =====================
GPU_ID=0
MEM_THRESHOLD=10
UTIL_THRESHOLD=10
# 替换为你的虚拟环境Python路径
CONDA_ENV_NAME="DrivingForward"
# 你的代码执行命令（激活环境后运行）
RUN_CMD=" CUDA_VISIBLE_DEVICES=0 python -W ignore train.py --weight_path results/main/20260314_212737/models/weights_1 --novel_view_mode SF"
CHECK_INTERVAL=10
# =================================================

echo "开始监控GPU $GPU_ID，每$CHECK_INTERVAL秒检测一次..."

while true; do
    # 修复：用更鲁棒的方式提取数值（适配所有nvidia-smi输出格式）
    # 1. 提取显存占用率（过滤掉非数字，只保留数值）
    
    mem_used=$(nvidia-smi --id=$GPU_ID --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    mem_total=$(nvidia-smi --id=$GPU_ID --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')
    # 2. 计算显存容量占用率（整数百分比）
    mem_usage=$(( 100 * mem_used / mem_total ))

    # 2. 提取GPU利用率（同上）
    gpu_util=$(nvidia-smi --id=$GPU_ID --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ' | grep -o '[0-9]*')

    # 兜底：若提取失败，默认设为100（视为被占用）
    if [ -z "$mem_usage" ]; then mem_usage=100; fi
    if [ -z "$gpu_util" ]; then gpu_util=100; fi
    
    # 打印当前状态
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] GPU $GPU_ID：显存占用率=$mem_usage%，GPU利用率=$gpu_util%"
    
    # 修复：用bash原生整数比较，避免bc的语法错误
    if [ $mem_usage -lt $MEM_THRESHOLD ] && [ $gpu_util -lt $UTIL_THRESHOLD ]; then
        echo "显卡空闲，开始执行代码！"
        conda activate $CONDA_ENV_NAME && $RUN_CMD
        echo "代码执行完成，退出监控。"
        exit 0
    fi
    
    sleep $CHECK_INTERVAL
done