"""
Minimal demo to load one batch of multi-scale OHLCV windows and a target
using the custom MultiScaleDataLoader and MultiScaleDataset.

Prerequisites:
- Feather files exist at user_data/data/{exchange}/{PAIR}-{tf}.feather
  e.g. user_data/data/okx/BTC_USDT-5m.feather
- scipy is installed (normalization uses scipy.stats.rankdata)
"""

from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from user_data.freqaimodels.ms_torch.data_loader import (
    DEFAULT_TIMEFRAME_CONFIGS,
    MultiScaleDataLoader,
)
from user_data.freqaimodels.ms_torch.dataset import (
    MultiScaleDataset,
    TargetGenerator,
    multi_scale_collate_fn,
)


def main() -> None:
    pair = "BTC/USDT"
    exchange = "okx"
    # 使用项目根目录定位数据,避免运行目录不同导致的相对路径错误
    repo_root = Path(__file__).resolve().parents[2]
    data_root = repo_root / "user_data" / "data"

    # 1) Init loader (loads all required timeframes into memory)
    loader = MultiScaleDataLoader(
        data_dir=data_root,
        pair=pair,
        timeframe_configs=DEFAULT_TIMEFRAME_CONFIGS,
        exchange=exchange,
    )

    # 2) Pick a valid timestamp (all timeframes have enough history)
    start_ts, _ = loader.get_valid_timestamp_range()
    ts = start_ts + pd.Timedelta(minutes=30)

    # 3) Generate training target: 48 future 5m candles (4h return, %)
    target_gen = TargetGenerator(loader, target_periods=48, target_timeframe="5m")
    targets = target_gen.generate_targets(pd.DatetimeIndex([ts]))

    # 4) Build dataset / dataloader
    dataset = MultiScaleDataset(
        timestamps=pd.DatetimeIndex([ts]),
        data_loader=loader,
        targets=targets,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=multi_scale_collate_fn,
    )

    # 5) Fetch one batch
    for x_dict, y in dataloader:
        print("5m shape:", x_dict["5m"].shape)  # torch.Size([1, 144, 6])
        print("1h shape:", x_dict["1h"].shape)  # torch.Size([1, 96, 6])
        print("target:", y)  # future 4h return (%)


if __name__ == "__main__":
    main()
