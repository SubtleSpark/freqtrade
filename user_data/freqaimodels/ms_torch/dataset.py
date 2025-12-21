"""
多时间尺度 K 线 PyTorch Dataset

封装 MultiScaleDataLoader 和归一化逻辑,
提供训练和预测所需的数据接口。
"""

import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .data_loader import MultiScaleDataLoader
from .normalization import normalize_with_offset


logger = logging.getLogger(__name__)


class MultiScaleDataset(Dataset):
    """
    多时间尺度 K 线数据集

    训练模式: 返回 (x_dict, y)
    预测模式: 返回 x_dict
    """

    def __init__(
        self,
        timestamps: pd.Series | pd.DatetimeIndex | np.ndarray,
        data_loader: MultiScaleDataLoader,
        targets: np.ndarray | pd.Series | None = None,
    ):
        # 转换 timestamps 为 pandas DatetimeIndex
        if isinstance(timestamps, pd.Series):
            self.timestamps = pd.DatetimeIndex(timestamps.values)
        elif isinstance(timestamps, np.ndarray):
            self.timestamps = pd.DatetimeIndex(timestamps)
        else:
            self.timestamps = timestamps

        # 确保时间戳是 UTC
        if self.timestamps.tzinfo is None:
            self.timestamps = self.timestamps.tz_localize("UTC")

        self.data_loader = data_loader

        # 处理 targets
        if targets is not None:
            if isinstance(targets, pd.Series):
                self.targets = targets.values.astype(np.float32)
            else:
                self.targets = np.asarray(targets, dtype=np.float32)
        else:
            self.targets = None

        # 验证长度一致性
        if self.targets is not None and len(self.targets) != len(self.timestamps):
            raise ValueError(
                f"Length mismatch: timestamps={len(self.timestamps)}, targets={len(self.targets)}"
            )

        logger.info(
            f"Created MultiScaleDataset with {len(self)} samples, "
            f"mode={'train' if self.targets is not None else 'predict'}"
        )

    def __len__(self) -> int:
        return len(self.timestamps)

    def __getitem__(
        self, idx: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor] | dict[str, torch.Tensor]:
        ts = self.timestamps[idx]

        # 1. 获取原始多尺度窗口
        try:
            raw_windows = self.data_loader.get_window(ts)
            current_price = self.data_loader.get_current_price(ts)
        except ValueError as e:
            logger.error(f"Failed to get window for timestamp {ts}: {e}")
            raise

        # 2. 归一化
        normalized = normalize_with_offset(raw_windows, current_price)

        # 3. 转换为 tensor
        x_dict = {
            tf: torch.from_numpy(arr)  # 已经是 float32
            for tf, arr in normalized.items()
        }

        if self.targets is not None:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return x_dict, y
        else:
            return x_dict


def multi_scale_collate_fn(batch: list):
    """
    自定义批处理函数:将多个样本的字典格式数据按时间周期堆叠。
    """
    if isinstance(batch[0], tuple):
        # 训练模式: [(x_dict, y), ...]
        x_dicts, ys = zip(*batch, strict=True)
        x_batched = {tf: torch.stack([x[tf] for x in x_dicts]) for tf in x_dicts[0].keys()}
        y_batched = torch.stack(ys)
        return x_batched, y_batched

    # 预测模式: [x_dict, ...]
    x_batched = {tf: torch.stack([x[tf] for x in batch]) for tf in batch[0].keys()}
    return x_batched


class TargetGenerator:
    """
    目标值生成器:根据未来 N 根 K 线计算收益率 (%)
    """

    def __init__(
        self,
        data_loader: MultiScaleDataLoader,
        target_periods: int = 48,
        target_timeframe: str = "5m",
    ):
        self.data_loader = data_loader
        self.target_periods = target_periods
        self.target_timeframe = target_timeframe

    def generate_targets(
        self,
        timestamps: pd.DatetimeIndex,
    ) -> np.ndarray:
        if timestamps.tzinfo is None:
            timestamps = timestamps.tz_localize("UTC")

        df = self.data_loader.data_cache[self.target_timeframe]
        targets = np.full(len(timestamps), np.nan, dtype=np.float32)

        for i, ts in enumerate(timestamps):
            try:
                current_idx = self.data_loader._locate_index(df, ts, self.target_timeframe)
                current_close = df.iloc[current_idx]["close"]

                future_idx = current_idx + self.target_periods
                if future_idx >= len(df):
                    continue

                future_close = df.iloc[future_idx]["close"]
                targets[i] = (future_close - current_close) / current_close * 100
            except ValueError:
                continue

        valid_count = np.sum(~np.isnan(targets))
        logger.info(
            f"Generated targets: {valid_count}/{len(timestamps)} valid "
            f"({valid_count / len(timestamps) * 100:.1f}%)"
        )

        return targets
