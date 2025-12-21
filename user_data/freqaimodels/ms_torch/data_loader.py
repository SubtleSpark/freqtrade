"""
多时间尺度 K 线数据加载器

从 feather 文件高效加载多个时间周期的 K 线数据,
实现时间对齐和窗口提取。
"""

import logging
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class TimeframeConfig(TypedDict):
    """单个时间周期的配置"""

    seq_len: int  # 窗口长度(K 线数量)
    patch_size: int  # Patch 压缩比例(模型侧使用,数据加载时不需要)


# 默认时间周期配置
DEFAULT_TIMEFRAME_CONFIGS: dict[str, TimeframeConfig] = {
    "1d": {"seq_len": 16, "patch_size": 1},
    "4h": {"seq_len": 48, "patch_size": 1},
    "1h": {"seq_len": 96, "patch_size": 4},
    "15m": {"seq_len": 96, "patch_size": 4},
    "5m": {"seq_len": 144, "patch_size": 6},
}

# 时间周期到 pandas Timedelta 的映射
TIMEFRAME_DELTAS = {
    "1m": pd.Timedelta(minutes=1),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}


class MultiScaleDataLoader:
    """
    高效的多时间尺度 K 线数据加载器

    特点:
    - 预加载 feather 文件到内存(内存换时间)
    - 实现 _locate_index() 时间对齐,避免未来数据泄露
    - 支持窗口数据提取

    Attributes:
        data_cache: 缓存的 K 线数据 {timeframe: DataFrame}
        timeframe_configs: 时间周期配置 {tf: {seq_len, patch_size}}
        pair: 交易对名称

    Example:
        >>> loader = MultiScaleDataLoader(
        ...     data_dir=Path("user_data/data"),
        ...     pair="BTC/USDT",
        ...     timeframe_configs=DEFAULT_TIMEFRAME_CONFIGS,
        ...     exchange="okx",
        ... )
        >>> windows = loader.get_window(pd.Timestamp("2024-01-15 14:35:00", tz="UTC"))
        >>> windows["5m"].shape  # (144, 5) - OHLCV
    """

    def __init__(
        self,
        data_dir: Path | str,
        pair: str,
        timeframe_configs: dict[str, TimeframeConfig] | None = None,
        exchange: str = "okx",
    ):
        """
        初始化数据加载器

        Args:
            data_dir: 数据目录路径(包含 exchange 子目录)
            pair: 交易对,如 "BTC/USDT"
            timeframe_configs: 时间周期配置,None 则使用默认配置
            exchange: 交易所名称,用于构建文件路径
        """
        self.data_dir = Path(data_dir)
        self.pair = pair
        self.exchange = exchange
        self.timeframe_configs = timeframe_configs or DEFAULT_TIMEFRAME_CONFIGS

        # 缓存: {tf: DataFrame}
        self.data_cache: dict[str, pd.DataFrame] = {}

        # 预加载所有时间周期的数据
        self._load_data()

    def _get_feather_path(self, timeframe: str) -> Path:
        """
        构建 feather 文件路径

        文件命名规范: {exchange}/{PAIR}-{timeframe}.feather
        例如: okx/BTC_USDT-5m.feather
        """
        pair_filename = self.pair.replace("/", "_")
        return self.data_dir / self.exchange / f"{pair_filename}-{timeframe}.feather"

    def _load_data(self) -> None:
        """预加载所有时间周期的 K 线数据到内存"""
        for tf in self.timeframe_configs:
            path = self._get_feather_path(tf)

            if not path.exists():
                raise FileNotFoundError(
                    f"Data file not found: {path}. Please download {tf} data for {self.pair} first."
                )

            logger.info(f"Loading {self.pair} {tf} data from {path}")

            df = pd.read_feather(path)

            # 确保 date 列是 UTC datetime
            df["date"] = pd.to_datetime(df["date"], utc=True)

            # 按时间排序并重置索引
            df = df.sort_values("date").reset_index(drop=True)

            self.data_cache[tf] = df
            logger.info(
                f"Loaded {len(df):,} candles for {self.pair} {tf}, "
                f"range: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}"
            )

    def _locate_index(
        self,
        df: pd.DataFrame,
        timestamp: pd.Timestamp,
        tf: str,
    ) -> int:
        """
        找到 timestamp 之前最近一根已完成的 K 线索引

        关键: 使用 K 线的收盘时间进行定位,确保不引入未来数据。
        K 线的收盘时间 = 开始时间 (date) + 时间周期 (tf_delta)

        Args:
            df: K 线 DataFrame,'date' 列是 K 线开始时间
            timestamp: 当前时间戳
            tf: 时间周期字符串,如 '5m', '1h', '1d'

        Returns:
            最新完成 K 线的索引(在 df 中的行号)

        Raises:
            ValueError: 如果没有找到已完成的 K 线

        Example:
            假设当前时间戳: 2024-01-15 14:35 UTC

            各周期最新完成的 K 线:
            - 5m:  14:30 - 14:35 (刚完成)
            - 1h:  13:00 - 14:00 (14:00-15:00 还在进行中)
            - 4h:  08:00 - 12:00 (12:00-16:00 还在进行中)
            - 1d:  01-14 00:00   (今天的日线还未收盘)
        """
        if tf not in TIMEFRAME_DELTAS:
            raise ValueError(f"Unknown timeframe: {tf}")

        tf_delta = TIMEFRAME_DELTAS[tf]

        # 计算每根 K 线的收盘时间
        close_times = df["date"] + tf_delta

        # 找到 close_time <= timestamp 的所有 K 线(已完成的)
        valid_mask = close_times <= timestamp

        if not valid_mask.any():
            raise ValueError(
                f"No completed {tf} candle before {timestamp}. "
                f"Data range: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}"
            )

        # 返回最后一个已完成 K 线的索引
        return df.index[valid_mask].max()

    def get_window(self, timestamp: pd.Timestamp) -> dict[str, np.ndarray]:
        """
        根据时间戳获取多尺度窗口数据

        Args:
            timestamp: 当前时间戳,必须是 UTC 时区

        Returns:
            多尺度 K 线窗口数据
            格式: {tf: np.ndarray[seq_len, 5]}
            列顺序: [open, high, low, close, volume]

        Raises:
            ValueError: 如果某个时间周期的历史数据不足
        """
        # 确保 timestamp 是 UTC
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        elif str(timestamp.tzinfo) != "UTC":
            timestamp = timestamp.tz_convert("UTC")

        result = {}

        for tf, config in self.timeframe_configs.items():
            seq_len = config["seq_len"]
            df = self.data_cache[tf]

            # 找到时间戳对应的位置(关键:避免未来数据泄露)
            idx = self._locate_index(df, timestamp, tf)

            # 检查是否有足够的历史数据
            if idx < seq_len - 1:
                raise ValueError(
                    f"Insufficient {tf} data for {self.pair}: "
                    f"need {seq_len} candles, have {idx + 1}. "
                    f"Timestamp: {timestamp}"
                )

            # 提取窗口 [idx-seq_len+1 : idx+1]
            start_idx = idx - seq_len + 1
            window = df.iloc[start_idx : idx + 1][["open", "high", "low", "close", "volume"]]

            result[tf] = window.values.astype(np.float32)

        return result

    def get_current_price(self, timestamp: pd.Timestamp) -> float:
        """
        获取当前价格(5m close[-1])

        这是归一化的基准价格,代表"当前市场价格"。

        Args:
            timestamp: 当前时间戳

        Returns:
            当前价格(5m 时间周期最新已完成 K 线的收盘价)
        """
        # 确保有 5m 数据
        if "5m" not in self.data_cache:
            # 如果没有 5m,使用最小时间周期
            smallest_tf = min(
                self.timeframe_configs.keys(),
                key=lambda x: TIMEFRAME_DELTAS[x],
            )
            df = self.data_cache[smallest_tf]
            idx = self._locate_index(df, timestamp, smallest_tf)
        else:
            df = self.data_cache["5m"]
            idx = self._locate_index(df, timestamp, "5m")

        return float(df.iloc[idx]["close"])

    def get_data_range(self, tf: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        """获取指定时间周期的数据范围"""
        df = self.data_cache[tf]
        return df["date"].iloc[0], df["date"].iloc[-1]

    def get_valid_timestamp_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """
        获取所有时间周期都有足够数据的有效时间范围

        Returns:
            (start_timestamp, end_timestamp) 可用于训练/预测的时间范围
        """
        # 计算每个时间周期需要的最早时间点
        earliest_valid = None

        for tf, config in self.timeframe_configs.items():
            seq_len = config["seq_len"]
            df = self.data_cache[tf]
            tf_delta = TIMEFRAME_DELTAS[tf]

            # 需要 seq_len 根 K 线,所以最早可用时间是第 seq_len 根 K 线的收盘时间
            if len(df) < seq_len:
                raise ValueError(f"Insufficient {tf} data: need {seq_len}, have {len(df)}")

            # 第 seq_len-1 索引(0-based)的 K 线收盘时间
            tf_earliest = df.iloc[seq_len - 1]["date"] + tf_delta

            if earliest_valid is None or tf_earliest > earliest_valid:
                earliest_valid = tf_earliest

        # 最晚可用时间:取所有时间周期数据的最小结束时间
        latest_valid = None
        for tf in self.timeframe_configs:
            df = self.data_cache[tf]
            tf_delta = TIMEFRAME_DELTAS[tf]
            tf_latest = df.iloc[-1]["date"] + tf_delta

            if latest_valid is None or tf_latest < latest_valid:
                latest_valid = tf_latest

        return earliest_valid, latest_valid
