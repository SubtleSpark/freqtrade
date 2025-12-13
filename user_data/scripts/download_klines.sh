#!/bin/bash

# K线数据下载脚本
# 用法: ./download_klines.sh

# ============ 配置 ============
# 交易对列表（手动维护）
PAIRS=("BTC/USDT" "ETH/USDT" "SOL/USDT" "DOGE/USDT" "XRP/USDT" "OKB/USDT")

# 时间周期
TIMEFRAMES=("1m" "5m" "30m" "1h" "4h")

# 时间范围配置
START_YEAR=2024
START_MONTH=1
END_YEAR=2024
END_MONTH=12

# 交易所
EXCHANGE="okx"

# ============ 脚本逻辑 ============
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# 激活虚拟环境
source "$PROJECT_DIR/.venv/bin/activate"

echo "开始下载K线数据..."
echo "交易所: $EXCHANGE"
echo "交易对: ${PAIRS[*]}"
echo "时间周期: ${TIMEFRAMES[*]}"
echo "时间范围: ${START_YEAR}年${START_MONTH}月 - ${END_YEAR}年${END_MONTH}月"
echo "================================"

# 按月循环下载（从后往前）
current_year=$END_YEAR
current_month=$END_MONTH

while true; do
    # 计算下个月（用于结束日期）
    next_month=$((current_month + 1))
    next_year=$current_year
    if [ $next_month -gt 12 ]; then
        next_month=1
        next_year=$((current_year + 1))
    fi

    # 格式化日期
    start_date=$(printf "%04d%02d01" $current_year $current_month)
    end_date=$(printf "%04d%02d01" $next_year $next_month)
    timerange="${start_date}-${end_date}"

    echo ""
    echo "======== ${current_year}年${current_month}月 ========"

    for PAIR in "${PAIRS[@]}"; do
        for TF in "${TIMEFRAMES[@]}"; do
            echo ">>> 下载 $PAIR $TF ($timerange) ..."
            freqtrade download-data \
                --exchange "$EXCHANGE" \
                --pairs "$PAIR" \
                -t "$TF" \
                --timerange "$timerange" \
                --prepend
        done
    done

    # 检查是否到达起始月份
    if [ $current_year -eq $START_YEAR ] && [ $current_month -eq $START_MONTH ]; then
        break
    fi

    # 移动到上个月
    current_month=$((current_month - 1))
    if [ $current_month -lt 1 ]; then
        current_month=12
        current_year=$((current_year - 1))
    fi
done

echo ""
echo "================================"
echo "下载完成!"
