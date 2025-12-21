"""
多时间尺度 K 线数据归一化模块

实现方案 B(自归一化 + offset)和方案 V3(成交量百分位排名)

输出维度: 6 = [O_norm, H_norm, L_norm, C_norm, V_rank, offset]
"""

import numpy as np
from scipy.stats import rankdata


def normalize_volume_percentile(volume: np.ndarray) -> np.ndarray:
    """
    将成交量转换为窗口内的百分位排名

    使用 scipy.stats.rankdata 实现,处理相同值的情况。
    """
    seq_len = len(volume)
    if seq_len <= 1:
        return np.zeros(seq_len)

    ranks = rankdata(volume, method="average")
    return (ranks - 1) / (seq_len - 1)


def normalize_with_offset(
    windows: dict[str, np.ndarray],
    current_price: float,
) -> dict[str, np.ndarray]:
    """
    多时间尺度 K 线归一化(方案 B + V3)

    每个时间尺度使用自己的 close[-1] 作为归一化基准,
    同时添加 offset 特征表示该尺度与当前价格的关系。
    """
    result = {}

    for tf, window in windows.items():
        seq_len = len(window)

        ohlc = window[:, :4]
        volume = window[:, 4]

        self_base = window[-1, 3]
        if self_base <= 0:
            raise ValueError(f"Invalid self_base={self_base} for timeframe {tf}")

        price_norm = ohlc / self_base
        vol_rank = normalize_volume_percentile(volume)
        offset = (self_base / current_price) - 1

        offset_col = np.full((seq_len, 1), offset, dtype=np.float32)
        vol_col = vol_rank[:, np.newaxis]

        result[tf] = np.concatenate([price_norm, vol_col, offset_col], axis=1).astype(np.float32)

    return result


def normalize_unified(
    windows: dict[str, np.ndarray],
    current_price: float,
) -> dict[str, np.ndarray]:
    """
    统一归一化方案(备选)
    所有时间尺度都用同一个基准价格(current_price)归一化。
    """
    result = {}

    for tf, window in windows.items():
        ohlc = window[:, :4]
        volume = window[:, 4]

        price_norm = ohlc / current_price

        vol_mean = volume.mean()
        if vol_mean > 0:
            vol_norm = volume / vol_mean
        else:
            vol_norm = np.ones_like(volume)

        result[tf] = np.column_stack([price_norm, vol_norm]).astype(np.float32)

    return result
