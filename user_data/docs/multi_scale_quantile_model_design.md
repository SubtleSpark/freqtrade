# 多时间尺度 Transformer 分位数预测模型 - 架构设计文档

> 版本: v0.2 (改进版)
> 日期: 2025-12
> 变更: 解决内存爆炸、过早池化、多列标签等工程问题

---

## 1. 项目目标

构建一个基于深度学习的量化交易预测系统，通过分析多个时间尺度的 K 线数据，输出未来价格变化的**概率分布**（而非单点预测），为交易决策提供更丰富的风险信息。

### 1.1 为什么选择概率分布预测？

| 方法 | 输出示例 | 优势 | 劣势 |
|------|----------|------|------|
| 单点预测 | "预测涨 2%" | 简单直接 | 无法量化不确定性 |
| **分位数预测** | "P10=-3%, P50=+1%, P90=+5%" | 可量化风险、支持仓位管理 | 模型稍复杂 |
| 完整分布 | μ=+1%, σ=3% (高斯) | 数学完整 | 假设分布形式，灵活性差 |

**选择分位数预测的原因**:
1. 不假设分布形式（可处理偏态、厚尾）
2. 直接可用于风险管理（VaR、CVaR）
3. 支持动态仓位调整（根据上下行空间比例）
4. 模型实现相对简单（使用 Pinball Loss）

---

## 2. 输入设计

### 2.1 多时间尺度 K 线输入 (精简版)

**设计原则**: 平衡信息量与计算效率，避免内存爆炸。

```
精简后的时间尺度配置:
┌─────────┬──────────┬───────────┬──────────────────────────────────┐
│ 周期    │ K线数量  │ Patch压缩 │ 最终 Token 数                    │
├─────────┼──────────┼───────────┼──────────────────────────────────┤
│ 1d      │ 16 根    │ 无        │ 16 tokens (保留完整序列)         │
│ 4h      │ 48 根    │ 无        │ 48 tokens (保留完整序列)         │
│ 1h      │ 96 根    │ 4:1       │ 24 tokens (每4根合并为1 patch)   │
│ 15m     │ 96 根    │ 4:1       │ 24 tokens (每4根合并为1 patch)   │
│ 5m      │ 144 根   │ 6:1       │ 24 tokens (每6根合并为1 patch)   │
└─────────┴──────────┴───────────┴──────────────────────────────────┘
总 Token 数: 16 + 48 + 24 + 24 + 24 = 136 tokens
(相比原方案 688 tokens 减少 80%)
```

**为什么要 Patch 压缩？**
1. 高频数据 (5m/15m) 相邻 K 线高度相关，压缩损失小
2. 减少 Transformer 的 O(n²) 注意力计算量
3. 保留关键时间结构信息

### 2.2 数据获取方式 (解决内存问题)

**❌ 原方案 (有问题)**:
```python
# 在策略中展开所有特征 → DataFrame 有 3440 列 → 内存爆炸
dataframe['%tf_1d_ohlcv_0_open'] = ...
dataframe['%tf_1d_ohlcv_0_high'] = ...
# ... 3440 个特征列
```

**✅ 改进方案: 模型内部数据加载**

```
数据流设计:
┌─────────────────────────────────────────────────────────────────────┐
│ 策略层 (Strategy)                                                   │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ 只提供少量元特征:                                               │ │
│ │   %meta_timestamp     # 当前时间戳                              │ │
│ │   %meta_pair          # 交易对编码                              │ │
│ │   %rsi_14, %ma_cross  # 可选的辅助指标 (少量)                   │ │
│ └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 模型层 (MultiScaleQuantileTransformer)                              │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ 自定义 DataLoader:                                              │ │
│ │   1. 根据 timestamp 定位到 feather 文件中的位置                 │ │
│ │   2. 直接从 feather 读取多时间尺度的原始 OHLCV                  │ │
│ │   3. 构建 {tf: [B, seq_len, 5]} 张量字典                        │ │
│ │   4. 在 GPU 上进行归一化                                        │ │
│ └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

**核心优势**:
- 策略 DataFrame 只有几列元数据，内存占用极小
- 模型按需加载数据，支持流式处理
- 可利用内存映射 (mmap) 高效读取 feather 文件

### 2.3 时间对齐与数据定位 (关键)

**问题**: 不同时间周期的"最新完成 K 线"时间点不同，必须正确处理以避免未来数据泄露。

```
假设当前时间戳: 2024-01-15 14:35 UTC

各周期最新完成的 K 线:
┌─────────┬─────────────────────┬──────────────────────────────┐
│ 周期    │ 最新完成 K 线时间   │ 说明                         │
├─────────┼─────────────────────┼──────────────────────────────┤
│ 5m      │ 14:30 - 14:35       │ 刚完成                       │
│ 15m     │ 14:15 - 14:30       │ 14:30-14:45 还在进行中       │
│ 1h      │ 13:00 - 14:00       │ 14:00-15:00 还在进行中       │
│ 4h      │ 08:00 - 12:00       │ 12:00-16:00 还在进行中       │
│ 1d      │ 01-14 00:00 UTC     │ 今天的日线还未收盘           │
└─────────┴─────────────────────┴──────────────────────────────┘
```

**时间定位函数实现**:

```python
def _locate_index(self, df: pd.DataFrame, timestamp: pd.Timestamp, tf: str) -> int:
    """
    找到 timestamp 之前最近一根已完成的 K 线索引

    关键: 使用 K 线的收盘时间进行定位，确保不引入未来数据

    Args:
        df: K 线 DataFrame，'date' 列是 K 线开始时间
        timestamp: 当前时间戳
        tf: 时间周期字符串，如 '5m', '1h', '1d'

    Returns:
        最新完成 K 线的索引
    """
    # 解析时间周期
    tf_delta = pd.Timedelta(tf)

    # 计算每根 K 线的收盘时间
    close_times = df['date'] + tf_delta

    # 找到 close_time <= timestamp 的最后一个 (已完成的 K 线)
    valid_mask = close_times <= timestamp
    if not valid_mask.any():
        raise ValueError(f"No completed {tf} candle before {timestamp}")

    return df.index[valid_mask].max()
```

### 2.4 数据缓存与窗口获取

```python
class MultiScaleDataLoader:
    """高效的多时间尺度数据加载器"""

    def __init__(self, data_dir: str, pair: str, timeframes: list):
        # 预加载数据到内存
        self.data_cache = {}
        for tf in timeframes:
            path = f"{data_dir}/{pair.replace('/', '_')}-{tf}.feather"
            self.data_cache[tf] = pd.read_feather(path)

        # 确保时间索引排序
        for tf in timeframes:
            self.data_cache[tf] = self.data_cache[tf].sort_values('date').reset_index(drop=True)

    def get_window(self, timestamp: pd.Timestamp, tf_configs: dict) -> dict:
        """
        根据时间戳获取多尺度窗口数据

        返回: {
            '1d': np.array([seq_len, 5]),   # OHLCV
            '4h': np.array([seq_len, 5]),
            ...
        }
        """
        result = {}
        for tf, config in tf_configs.items():
            seq_len = config['seq_len']
            df = self.data_cache[tf]

            # 找到时间戳对应的位置 (关键: 避免未来数据泄露)
            idx = self._locate_index(df, timestamp, tf)

            # 检查是否有足够的历史数据
            if idx < seq_len - 1:
                raise ValueError(f"Insufficient {tf} data: need {seq_len}, have {idx + 1}")

            # 提取窗口 [idx-seq_len+1 : idx+1]
            window = df.iloc[idx-seq_len+1:idx+1][['open', 'high', 'low', 'close', 'volume']]
            result[tf] = window.values

        return result
```

### 2.5 数据预处理与归一化

#### 2.5.1 归一化的核心问题

不同时间尺度的 `close[-1]` 对应不同时间点的价格：

```
假设当前 BTC 价格 = $100,000 (5m close[-1])

各尺度的 close[-1]:
┌─────────┬─────────────┬──────────────────────────────┐
│ 周期    │ close[-1]   │ 对应时间                     │
├─────────┼─────────────┼──────────────────────────────┤
│ 5m      │ $100,000    │ 14:35 (当前)                 │
│ 1h      │ $99,500     │ 14:00 (1小时前)              │
│ 4h      │ $98,500     │ 12:00 (2.5小时前)            │
│ 1d      │ $98,000     │ 昨天收盘                     │
└─────────┴─────────────┴──────────────────────────────┘
```

**问题**: 如果每个尺度用自己的 `close[-1]` 归一化，跨尺度注意力时价格值不可直接比较。

#### 2.5.2 方案 A: 统一归一化基准

**思路**: 所有尺度都用 5m 的 `close[-1]`（当前真实价格）作为归一化基准。

```python
def normalize_unified(windows: dict, current_price: float) -> dict:
    """
    统一归一化: 所有尺度用同一个基准价格

    Args:
        windows: {tf: np.array([seq_len, 5])}  # OHLCV
        current_price: 5m close[-1]，当前市场价格

    Returns:
        {tf: np.array([seq_len, 5])}  # 归一化后的 OHLCV
    """
    result = {}
    for tf, window in windows.items():
        normalized = window.copy()
        # 价格列 (OHLC) 归一化
        normalized[:, :4] = window[:, :4] / current_price
        # 成交量归一化 (相对窗口均值)
        vol_mean = window[:, 4].mean()
        normalized[:, 4] = window[:, 4] / vol_mean if vol_mean > 0 else 1.0
        result[tf] = normalized
    return result
```

**示例**:
```
1d 窗口 (16天前 ~ 昨天), 基准 = $100,000:

K线[0]  (16天前): O=0.920, H=0.935, L=0.910, C=0.930  # 远离 1.0
K线[15] (昨天):   O=0.975, H=0.990, L=0.970, C=0.980  # 接近 1.0
```

| 优点 | 缺点 |
|------|------|
| ✅ 跨尺度价格直接可比较 | ❌ 数值范围大 (0.5-1.5) |
| ✅ 实现简单 | ❌ 可能过拟合绝对价格水平 |
| ✅ 趋势信息隐含在数值中 | ❌ 不同币种波动率差异大 |

**适用场景**: 单一币种、价格波动较稳定的情况

#### 2.5.3 方案 B: 自身归一化 + 跨尺度偏移 (推荐)

**思路**: 每个尺度用自己的 `close[-1]` 归一化（保持形态），同时添加一个 `offset` 特征表示与当前价格的关系。

```python
def normalize_with_offset(windows: dict, current_price: float) -> dict:
    """
    自身归一化 + 偏移: 保持形态，显式传递跨尺度信息

    Args:
        windows: {tf: np.array([seq_len, 5])}  # OHLCV
        current_price: 5m close[-1]，当前市场价格

    Returns:
        {tf: np.array([seq_len, 6])}  # [O_norm, H_norm, L_norm, C_norm, V_norm, offset]
    """
    result = {}
    for tf, window in windows.items():
        seq_len = len(window)

        # 1. 自身基准 (该尺度的 close[-1])
        self_base = window[-1, 3]  # close of last candle

        # 2. 价格归一化 (相对自身基准)
        price_norm = window[:, :4] / self_base  # OHLC, 围绕 1.0

        # 3. 成交量归一化 (相对窗口均值)
        vol_mean = window[:, 4].mean()
        vol_norm = window[:, 4] / vol_mean if vol_mean > 0 else np.ones(seq_len)

        # 4. 跨尺度偏移 (该尺度基准相对于当前价格)
        offset = (self_base / current_price) - 1  # 标量
        offset_col = np.full((seq_len, 1), offset)

        # 5. 组合: [seq_len, 6]
        result[tf] = np.concatenate([
            price_norm,           # [seq_len, 4]
            vol_norm[:, None],    # [seq_len, 1]
            offset_col            # [seq_len, 1]
        ], axis=1)

    return result
```

**示例**:
```
场景: 当前价格 $100,000, 日线基准 $98,000 (昨天收盘)

1d 窗口归一化后 (基准 = $98,000, offset = -0.02):

K线[0]  (16天前): O=0.939, H=0.954, L=0.929, C=0.949, V=0.7, offset=-0.02
K线[15] (昨天):   O=0.995, H=1.010, L=0.990, C=1.000, V=1.0, offset=-0.02
                                              ↑                    ↑
                                          围绕 1.0              告诉模型:
                                          形态保留              "日线基准比现价低 2%"

5m 窗口归一化后 (基准 = $100,000, offset = 0.00):

K线[0]  (12小时前): O=0.985, H=0.986, L=0.984, C=0.986, V=1.2, offset=0.00
K线[143] (现在):    O=0.999, H=1.001, L=0.998, C=1.000, V=1.1, offset=0.00
```

| 优点 | 缺点 |
|------|------|
| ✅ 数值稳定 (0.9-1.1 范围) | ❌ 特征维度增加 (5→6) |
| ✅ 保留 K 线形态特征 | ❌ offset 是标量，信息密度低 |
| ✅ 跨尺度信息通过 offset 显式传递 | |
| ✅ 泛化能力更好 (形态独立于绝对价格) | |

**模型如何利用 offset**:
```
跨尺度注意力时，模型可以学习:

1. 形态识别 (来自 OHLC_norm):
   - 5m 出现连续阳线
   - 1d 形成上升趋势

2. 趋势强度 (来自 offset):
   - offset_1d = -0.02 → 过去 16 天涨了 2%
   - offset_4h = -0.015 → 过去 8 天涨了 1.5%
   - 模型理解: 中长期趋势向上

3. 跨尺度对比 (通过 offset 差异):
   - |offset_1d - offset_5m| = 0.02
   - 模型判断: 短期涨幅相对长期是否过快/过慢
```

**适用场景**: 多币种、需要泛化能力、不同市场状态

#### 2.5.4 价格归一化方案选择

```
┌─────────────┬────────────────┬────────────────┐
│             │ 方案 A         │ 方案 B (推荐)  │
│             │ 统一基准       │ 自身+偏移      │
├─────────────┼────────────────┼────────────────┤
│ 数值范围    │ 0.5 ~ 1.5      │ 0.9 ~ 1.1      │
│ 实现复杂度  │ 简单           │ 中等           │
│ 形态保留    │ 部分           │ 完整           │
│ 跨尺度信息  │ 隐式           │ 显式 (offset)  │
│ 泛化能力    │ 较差           │ 较好           │
│ 推荐场景    │ 单币种原型     │ 生产环境       │
└─────────────┴────────────────┴────────────────┘
```

#### 2.5.5 成交量归一化 (独立处理)

成交量与价格有本质区别，需要单独设计归一化方案。

**成交量的特殊性**:
```
1. 不同时间尺度的成交量不可直接比较:
   - 5m  均值: 100 BTC/根
   - 1h  均值: 1,200 BTC/根 (约 12×5m)
   - 1d  均值: 28,800 BTC/根

2. 分布特点:
   - 高度右偏 (大部分时间低，偶尔暴涨)
   - 跨币种差异巨大 (BTC vs 山寨币)
   - 有明显的日内/周内周期性
```

##### 方案 V1: 窗口均值归一化 (简单)

```python
def normalize_volume_mean(volume: np.ndarray) -> np.ndarray:
    """相对于窗口均值归一化"""
    vol_mean = volume.mean()
    return volume / vol_mean if vol_mean > 0 else np.ones_like(volume)

# 示例
volume = [100, 50, 200, 80, 150]  # 均值 = 116
vol_norm = [0.86, 0.43, 1.72, 0.69, 1.29]  # 围绕 1.0
```

| 优点 | 缺点 |
|------|------|
| ✅ 实现简单 | ❌ 跨尺度不可比 |
| ✅ 围绕 1.0 波动 | ❌ 异常值影响均值 |
|  | ❌ 数值范围不固定 (0~∞) |

**适用**: 快速原型、单一时间尺度

##### 方案 V2: 对数变换 + Z-score 标准化

```python
def normalize_volume_log_zscore(volume: np.ndarray) -> np.ndarray:
    """对数变换后 Z-score 标准化"""
    # 对数变换压缩极端值
    vol_log = np.log1p(volume)  # log(1+x) 避免 log(0)

    # Z-score 标准化
    mean, std = vol_log.mean(), vol_log.std()
    return (vol_log - mean) / std if std > 0 else np.zeros_like(vol_log)

# 示例
volume = [100, 50, 200, 80, 150]
vol_log = [4.62, 3.93, 5.30, 4.39, 5.02]
vol_norm = [-0.23, -1.50, 1.02, -0.65, 0.49]  # 围绕 0, 标准差 1
```

| 优点 | 缺点 |
|------|------|
| ✅ 压缩极端值 | ❌ 跨尺度仍不可比 |
| ✅ 更接近正态分布 | ❌ 需要记录 mean/std 用于推理 |
| ✅ 数值范围稳定 (-3~3) | |

**适用**: 单币种训练、对分布敏感的模型

##### 方案 V3: 百分位排名 (推荐原型)

```python
def normalize_volume_percentile(volume: np.ndarray) -> np.ndarray:
    """
    将成交量转换为窗口内的百分���排名 [0, 1]

    优势: 完全消除量纲，专注于"相对高低"
    """
    from scipy.stats import rankdata
    ranks = rankdata(volume, method='average')
    return (ranks - 1) / (len(ranks) - 1)  # 归一化到 [0, 1]

# 示例
volume = [100, 50, 200, 80, 150]
vol_norm = [0.50, 0.00, 1.00, 0.25, 0.75]
#           ↑     ↑     ↑     ↑     ↑
#          中等  最低  最高  较低  较高
```

| 优点 | 缺点 |
|------|------|
| ✅ 完全无量纲 [0, 1] | ❌ 丢失绝对大小信息 |
| ✅ 对异常值鲁棒 | ❌ 计算稍慢 (需要排序) |
| ✅ **跨尺度语义一致** | |

**语义一致性说明**:
```
5m  vol_rank = 0.9 → "这根 5m K线成交量在最近 144 根中排前 10%"
1d  vol_rank = 0.9 → "这根日线成交量在最近 16 天中排前 10%"

虽然绝对量不同，但"相对活跃度"语义相同，跨尺度注意力可以比较。
```

**适用**: 快速原型、跨尺度模型

##### 方案 V4: 双特征 (推荐生产)

**结合相对量和绝对量信息**:

```python
def normalize_volume_dual(
    volume: np.ndarray,
    global_median: float
) -> np.ndarray:
    """
    返回两个特征:
    1. 窗口内百分位排名 (相对位置) [0, 1]
    2. 相对于全局中位数的对数比例 (绝对水平)

    Args:
        volume: 窗口内成交量 [seq_len]
        global_median: 该币种该周期的历史中位数成交量

    Returns:
        [seq_len, 2] - [rank, global_ratio]
    """
    from scipy.stats import rankdata
    seq_len = len(volume)

    # 特征1: 窗口内排名 [0, 1]
    ranks = rankdata(volume, method='average')
    rank_norm = (ranks - 1) / (seq_len - 1) if seq_len > 1 else np.zeros(seq_len)

    # 特征2: 相对全局中位数的对数比例
    # log1p 使得: 中位数对应约 0.69, 2倍中位数约 1.10, 0.5倍约 0.41
    global_ratio = np.log1p(volume / global_median)

    return np.stack([rank_norm, global_ratio], axis=-1)  # [seq_len, 2]

# 示例
volume = [100, 50, 200, 80, 150]
global_median = 120  # 历史中位数

# 特征1: 窗口排名
rank_norm = [0.50, 0.00, 1.00, 0.25, 0.75]

# 特征2: 全局比例 (log1p(v/120))
global_ratio = [0.56, 0.29, 0.98, 0.41, 0.81]

# 组合输出: [5, 2]
vol_features = [
    [0.50, 0.56],  # K线0: 窗口中等，略低于历史中位数
    [0.00, 0.29],  # K线1: 窗口最低，明显低于历史
    [1.00, 0.98],  # K线2: 窗口最高，也是历史较高
    [0.25, 0.41],  # K线3: 窗口较低，低于历史
    [0.75, 0.81],  # K线4: 窗口较高，略高于历史
]
```

| 优点 | 缺点 |
|------|------|
| ✅ 同时保留相对和绝对信息 | ❌ 特征维度 +1 |
| ✅ 跨尺度可比 (排名语义一致) | ❌ 需要预计算全局中位数 |
| ✅ 对异常值鲁棒 | |
| ✅ 模型可学习"绝对放量"和"相对放量" | |

**全局中位数的获取**:
```python
# 训练前预计算并存储
global_volume_medians = {
    'BTC/USDT': {
        '5m': 150.0,
        '1h': 1800.0,
        '1d': 43200.0,
    },
    'ETH/USDT': {
        '5m': 500.0,
        # ...
    }
}
```

**适用**: 生产环境、多币种、需要精细成交量分析

##### 成交量方案对比总结

```
┌─────────────┬───────────┬───────────┬───────────┬─────────────┐
│             │ V1 均值   │ V2 对数   │ V3 排名   │ V4 双特征   │
│             │           │ +Z-score  │ (推荐原型)│ (推荐生产)  │
├─────────────┼───────────┼───────────┼───────────┼─────────────┤
│ 数值范围    │ 0~∞       │ -3~3      │ 0~1       │ 0~1 + log   │
│ 异常值鲁棒  │ ❌        │ ✅        │ ✅✅      │ ✅✅        │
│ 跨尺度可比  │ ❌        │ ❌        │ ✅        │ ✅          │
│ 保留绝对量  │ ✅        │ ✅        │ ❌        │ ✅          │
│ 实现复杂度  │ 简单      │ 中等      │ 中等      │ 较复杂      │
│ 额外特征维度│ 0         │ 0         │ 0         │ +1          │
└─────────────┴───────────┴───────────┴───────────┴─────────────┘
```

##### 推荐组合

```
组合1: 价格方案B + 成交量V3 (原型开发)
────────────────────────────────────────
特征: [O_norm, H_norm, L_norm, C_norm, V_rank, offset]
维度: 6
优点: 简单，跨尺度一致

组合2: 价格方案B + 成交量V4 (生产环境)
────────────────────────────────────────
特征: [O_norm, H_norm, L_norm, C_norm, V_rank, V_global, offset]
维度: 7
优点: 信息完整，可学习绝对/相对放量
```

#### 2.5.6 完整预处理流程

```
原始 OHLCV 窗口 (多尺度)
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  1. 时间对齐 (_locate_index)                        │
│     确保每个尺度只使用已完成的 K 线                 │
├─────────────────────────────────────────────────────┤
│  2. 提取窗口                                        │
│     各尺度按配置提取 seq_len 根 K 线                │
├─────────────────────────────────────────────────────┤
│  3. 价格归一化 (方案 A 或 B)                        │
│     A: 统一基准                                     │
│     B: 自身+偏移 (推荐)                             │
├─────────────────────────────────────────────────────┤
│  4. 成交量归一化 (方案 V1-V4)                       │
│     V3: 排名 (原型)                                 │
│     V4: 双特征 (生产)                               │
├─────────────────────────────────────────────────────┤
│  5. 对数变换 (可选，仅价格)                         │
│     price_log = log(price_norm)                     │
├─────────────────────────────────────────────────────┤
│  6. 转换为 Tensor                                   │
│     在 GPU 上执行后续计算                           │
└─────────────────────────────────────────────────────┘
       │
       ▼
 归一化后的多尺度 K 线张量

 组合1 (B+V3): {tf: [B, seq_len, 6]}
   → [O_norm, H_norm, L_norm, C_norm, V_rank, offset]

 组合2 (B+V4): {tf: [B, seq_len, 7]}
   → [O_norm, H_norm, L_norm, C_norm, V_rank, V_global, offset]

---

## 3. 输出设计

### 3.1 分位数输出

```
模型输出: 7 个分位数值
┌────────┬────────────────────────────────────────────┐
│ 分位数 │ 含义                                       │
├────────┼────────────────────────────────────────────┤
│ P5     │ 5% 概率跌破此值 (极端悲观)                 │
│ P10    │ 10% 概率跌破 (可用于止损)                  │
│ P25    │ 25% 概率跌破 (下四分位)                    │
│ P50    │ 中位数预测 (最可能的结果)                  │
│ P75    │ 75% 概率不超过 (上四分位)                  │
│ P90    │ 90% 概率不超过 (可用于止盈)                │
│ P95    │ 95% 概率不超过 (极端乐观)                  │
└────────┴────────────────────────────────────────────┘

输出单位: 百分比收益率
例如: P50 = +1.5 表示中位数预测未来上涨 1.5%
```

### 3.2 预测目标

```
预测目标: 未来 N 个周期的价格变化率

target = (future_close - current_close) / current_close × 100%

推荐目标:
- 点对点收益率: close[t+48] / close[t] - 1  (未来 4 小时)
```

**预测时间范围**: 4 小时 (对应 48 根 5 分钟 K 线)

### 3.3 训练目标 vs 模型输出

**关键理解**: 分位数预测只需要 **1 个训练目标**，但输出 **7 列预测**。

```
训练时:
┌───────────────┐     ┌─────────────────┐     ┌───────────────┐
│ 输入特征      │ ──▶ │ 模型            │ ──▶ │ 7 个分位数    │
│ X: [B, ...]   │     │                 │     │ ŷ: [B, 7]     │
└───────────────┘     └─────────────────┘     └───────┬───────┘
                                                      │
                                                      ▼
┌───────────────┐                            ┌───────────────────┐
│ 真实目标      │ ────────────────────────▶  │ Pinball Loss      │
│ y: [B, 1]     │  (广播到 7 个分位数)       │ 对每个分位数计算  │
└───────────────┘                            └───────────────────┘

损失计算:
loss = Σ pinball_loss(y, ŷ[:, i], tau=quantiles[i])
```

---

## 4. 模型架构 (改进版)

### 4.1 整体架构图 (延迟池化)

**核心改进**: 不在每个尺度编码器后立即池化，而是保留序列维度，让跨尺度注意力在 token 级别交互。

```
                              输入层
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
   ┌─────────┐            ┌─────────┐            ┌─────────┐
   │ 1d 编码 │            │ 4h 编码 │    ...     │ 5m 编码 │
   │  器     │            │  器     │            │  器     │
   └────┬────┘            └────┬────┘            └────┬────┘
        │                      │                      │
        │   Transformer        │   Transformer        │   Patch Conv
        │   Encoder            │   Encoder            │   + Transformer
        │                      │                      │
        ▼                      ▼                      ▼
   [B, 16, d]             [B, 48, d]             [B, 24, d]
        │                      │                      │
        │    ╔═══════════════════════════════════════╗
        └───▶║     Token 级别跨尺度注意力           ║◀───┘
             ║     Cross-Scale Token Attention       ║
             ║     (136 tokens 相互关注)             ║
             ╚═══════════════════╤═══════════════════╝
                                 │
                                 ▼
                    ┌──────────────────────┐
                    │  全局池化            │
                    │  [B, 136, d] → [B, d]│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  分位数输出头        │
                    │  Quantile Heads      │
                    └──────────┬───────────┘
                               │
                               ▼
                    [P5, P10, P25, P50, P75, P90, P95]
```

### 4.2 各组件详细设计

#### 4.2.1 低频尺度编码器 (1d, 4h) - 保留完整序列

```
输入: [batch, seq_len, D_in]
      D_in = 5 (方案A: OHLCV) 或 6 (方案B: OHLCV + offset)
       │
       ▼
┌─────────────────────────────────────────┐
│ Linear Projection: D_in → d_model       │
└─────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│ Learnable Positional Encoding           │
│ + Timeframe Type Embedding              │
│ (区分 1d 和 4h 的 token)                │
└─────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│ Transformer Encoder (×2 层)             │
│ - Multi-Head Self-Attention             │
│ - Feed-Forward Network                  │
│ - Pre-LN (更稳定)                       │
└─────────────────────────────────────────┘
       │
       ▼
输出: [batch, seq_len, d_model]  # 保留完整序列！
```

#### 4.2.2 高频尺度编码器 (1h, 15m, 5m) - Patch 压缩

```
输入: [batch, seq_len, D_in]  # 例如 5m: [B, 144, D_in]
      D_in = 5 (方案A) 或 6 (方案B)
       │
       ▼
┌─────────────────────────────────────────┐
│ Patch Embedding (1D Conv)               │
│                                         │
│ Conv1d(in=D_in, out=d_model, kernel=6, stride=6)
│ 将 6 根 K 线压缩为 1 个 patch           │
│ [B, 144, D_in] → [B, 24, d_model]       │
└─────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│ Positional Encoding + Type Embedding    │
└─────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│ Transformer Encoder (×2 层)             │
└─────────────────────────────────────────┘
       │
       ▼
输出: [batch, 24, d_model]  # 压缩后的 patch 序列
```

**Patch 压缩的优势**:
1. 保留局部时间结构 (卷积感受野)
2. 减少 token 数量，降低注意力计算量
3. 每个 patch 仍然有位置编码，保留时序信息

#### 4.2.3 跨尺度 Token 注意力 (核心改进)

```
输入: 5 个时间尺度的 token 序列 (未池化)
      1d:  [B, 16, d]
      4h:  [B, 48, d]
      1h:  [B, 24, d]  (patch)
      15m: [B, 24, d]  (patch)
      5m:  [B, 24, d]  (patch)
       │
       ▼
┌─────────────────────────────────────────────────────┐
│ 1. 拼接所有 token                                   │
│    [B, 16+48+24+24+24, d] = [B, 136, d]            │
├─────────────────────────────────────────────────────┤
│ 2. 添加时间尺度类型 Embedding                       │
│    type_emb = Embedding(5, d)  # 5 种尺度          │
│    tokens[:, 0:16] += type_emb[0]   # 1d tokens   │
│    tokens[:, 16:64] += type_emb[1]  # 4h tokens   │
│    ...                                             │
├─────────────────────────────────────────────────────┤
│ 3. Cross-Scale Transformer (×2 层)                 │
│    所有 136 个 token 相互做 attention              │
│    - 1d token 可以关注 5m patch                    │
│    - 5m patch 可以关注对应时间的 4h token          │
└─────────────────────────────────────────────────────┘
       │
       ▼
输出: [B, 136, d]  # 融合后的 token 表示
```

#### 4.2.4 全局池化与输出头

```
输入: [B, 136, d]  # 所有 token
       │
       ▼
┌─────────────────────────────────────────┐
│ Attention Pooling (推荐)                │
│                                         │
│ query = learnable_vector [1, d]         │
│ keys = tokens [B, 136, d]               │
│ attn = softmax(query @ keys.T)          │
│ output = attn @ tokens  →  [B, d]       │
│                                         │
│ (让模型学习关注哪些 token 最重要)       │
└─────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│ 分位数输出头                            │
│                                         │
│ MLP: [B, d] → [B, d//2] → [B, 7]       │
│ (共享骨干，隐式单调约束)               │
└─────────────────────────────────────────┘
       │
       ▼
输出: [B, 7]  # 7 个分位数预测
```

### 4.3 模型超参数

```python
model_config = {
    # 时间尺度配置
    'timeframe_configs': {
        '1d':  {'seq_len': 16,  'patch_size': 1},   # 无压缩
        '4h':  {'seq_len': 48,  'patch_size': 1},   # 无压缩
        '1h':  {'seq_len': 96,  'patch_size': 4},   # 4:1 压缩
        '15m': {'seq_len': 96,  'patch_size': 4},   # 4:1 压缩
        '5m':  {'seq_len': 144, 'patch_size': 6},   # 6:1 压缩
    },

    # 归一化方案 (见 2.5 节)
    'price_normalization': 'offset',   # 'unified' (方案A) 或 'offset' (方案B，推荐)
    'volume_normalization': 'rank',    # 'mean'(V1), 'zscore'(V2), 'rank'(V3), 'dual'(V4)

    # 输入特征维度 (根据归一化方案自动计算)
    # 方案 A + V1/V2/V3: d_input = 5
    # 方案 B + V3:       d_input = 6  [OHLC_norm, V_rank, offset]
    # 方案 B + V4:       d_input = 7  [OHLC_norm, V_rank, V_global, offset]
    'd_input': 6,  # 推荐: 方案B + V3

    # 模型结构
    'd_model': 128,
    'nhead': 8,
    'num_encoder_layers': 2,      # 每个尺度的 Transformer 层数
    'num_cross_scale_layers': 2,  # 跨尺度 Transformer 层数
    'dim_feedforward': 256,
    'dropout': 0.1,

    # 输出配置
    'quantiles': [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
}
```

---

## 5. 损失函数

### 5.1 Pinball Loss (分位数损失)

```
对于分位数 τ ∈ (0, 1):

L_τ(y, ŷ) = {
    τ × (y - ŷ)       如果 y ≥ ŷ  (真实值在预测之上)
    (1-τ) × (ŷ - y)   如果 y < ŷ  (真实值在预测之下)
}

等价形式:
L_τ(y, ŷ) = (y - ŷ) × (τ - I(y < ŷ))
```

**PyTorch 实现**:
```python
class QuantileLoss(nn.Module):
    def __init__(self, quantiles: list[float]):
        super().__init__()
        self.quantiles = torch.tensor(quantiles)  # [0.05, 0.10, ...]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: [B, num_quantiles], target: [B] or [B, 1]
        target = target.view(-1, 1)  # [B, 1]
        quantiles = self.quantiles.to(pred.device)  # [num_quantiles]

        errors = target - pred  # [B, num_quantiles]
        loss = torch.max(
            quantiles * errors,
            (quantiles - 1) * errors
        )
        return loss.mean()
```

### 5.2 单调性正则化

```python
def monotonic_penalty(pred: torch.Tensor) -> torch.Tensor:
    """确保 P5 < P10 < ... < P95"""
    # pred: [B, 7] 按分位数排序
    diff = pred[:, 1:] - pred[:, :-1]  # [B, 6]
    penalty = torch.relu(-diff).sum()  # 惩罚递减的情况
    return penalty
```

### 5.3 总损失

```python
total_loss = quantile_loss + lambda_mono * monotonic_penalty
# lambda_mono = 0.1 (经验值)
```

---

## 6. FreqAI 集成方案 (重点改进)

### 6.1 核心挑战与解决方案

| 挑战 | 解决方案 |
|------|----------|
| 策略展开特征导致内存爆炸 | 模型内部加载数据，策略只提供元信息 |
| FreqAI 默认单列标签 | 重写 `predict()` 返回多列 DataFrame |
| `label_pipeline` 只处理单列 | 跳过默认逆变换，手动处理 |
| `historic_predictions` 存储 | 自定义多列存储逻辑 |

### 6.2 继承结构

```
IFreqaiModel (FreqAI 接口)
     │
     ▼
BasePyTorchModel (PyTorch 基类)
     │
     ▼
MultiScaleQuantileTransformer (自定义模型)
     │
     ├── 重写 data_convertor (自定义数据加载)
     ├── 重写 fit() (自定义训练循环)
     └── 重写 predict() (多列输出)
```

### 6.3 策略层实现 (极简)

```python
class MultiScaleQuantileStrategy(IStrategy):
    """策略只提供元信息，不展开 K 线特征"""

    minimal_roi = {"0": 0.1}
    stoploss = -0.05
    timeframe = '5m'

    # 声明需要的时间周期 (FreqAI 会确保数据可用)
    informative_pairs = []  # 不使用 informative pairs

    def feature_engineering_standard(self, df: DataFrame, metadata: dict) -> DataFrame:
        """只添加少量元特征"""
        # 元特征: 让模型知道当前时间和交易对
        df['%meta_hour'] = df['date'].dt.hour
        df['%meta_dayofweek'] = df['date'].dt.dayofweek

        # 可选: 添加少量辅助指标
        df['%rsi_14'] = ta.RSI(df, timeperiod=14)
        df['%ma_cross'] = (ta.SMA(df, 20) > ta.SMA(df, 50)).astype(int)

        return df

    def set_freqai_targets(self, df: DataFrame, metadata: dict) -> DataFrame:
        """定义单一预测目标"""
        # 未来 4 小时收益率 (48 根 5m K 线)
        df['&-target'] = (
            df['close'].shift(-48) / df['close'] - 1
        ) * 100  # 百分比

        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """使用分位数预测进行入场"""
        # 模型返回的列: &-q5, &-q10, &-q25, &-q50, &-q75, &-q90, &-q95

        long_conditions = [
            df['do_predict'] == 1,
            df['&-q50'] > 0.5,  # 中位数预测上涨 0.5%
            df['&-q75'] / (abs(df['&-q25']) + 0.01) > 2,  # 风险收益比
            (df['&-q90'] - df['&-q10']) < 5,  # 不确定性不太大
        ]

        if long_conditions:
            df.loc[reduce(lambda x, y: x & y, long_conditions), 'enter_long'] = 1

        return df
```

### 6.4 模型类实现 (关键)

```python
class MultiScaleQuantileTransformer(BasePyTorchModel):
    """
    多时间尺度分位数预测模型

    关键改动:
    1. 自定义 data_convertor 从文件加载多尺度数据
    2. 重写 fit() 使用自定义 DataLoader
    3. 重写 predict() 返回多列分位数
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.quantiles = self.freqai_info.get(
            "model_training_parameters", {}
        ).get("quantiles", [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])

        # 分位数列名
        self.quantile_columns = [f"&-q{int(q*100)}" for q in self.quantiles]

    @property
    def data_convertor(self) -> PyTorchDataConvertor:
        """返回自定义数据转换器"""
        return MultiScaleDataConvertor(
            data_dir=self.config.get("user_data_dir", "user_data") + "/data",
            pair=self.pair,
            timeframe_configs=self.freqai_info.get(
                "model_training_parameters", {}
            ).get("timeframe_configs", {}),
        )

    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """训练模型"""
        # 1. 创建自定义 Dataset 和 DataLoader
        train_dataset = MultiScaleDataset(
            timestamps=data_dictionary["train_features"]['date'],
            targets=data_dictionary["train_labels"]["&-target"].values,
            data_loader=self.data_convertor.data_loader,
            timeframe_configs=self.timeframe_configs,
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # 2. 初始化模型
        model = MultiScaleQuantileTransformerModel(
            timeframe_configs=self.timeframe_configs,
            quantiles=self.quantiles,
            **self.model_kwargs
        ).to(self.device)

        # 3. 训练循环
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = QuantileLoss(self.quantiles)

        for epoch in range(self.n_epochs):
            for batch in train_loader:
                x_dict, y = batch
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                y = y.to(self.device)

                pred = model(x_dict)
                loss = criterion(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return model

    def predict(
        self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs
    ) -> tuple[DataFrame, NDArray[np.int_]]:
        """
        生成预测 - 返回多列分位数

        注意: 这里绕过了 FreqAI 默认的单列处理逻辑
        """
        # 1. 获取预测时间戳
        timestamps = unfiltered_df['date'].values

        # 2. 创建预测 Dataset
        pred_dataset = MultiScaleDataset(
            timestamps=timestamps,
            targets=None,  # 预测时没有目标
            data_loader=self.data_convertor.data_loader,
            timeframe_configs=self.timeframe_configs,
        )
        pred_loader = DataLoader(pred_dataset, batch_size=64, shuffle=False)

        # 3. 模型推理
        self.model.model.set_mode_for_inference()
        all_preds = []

        with torch.no_grad():
            for batch in pred_loader:
                x_dict = {k: v.to(self.device) for k, v in batch.items()}
                pred = self.model.model(x_dict)  # [B, 7]
                all_preds.append(pred.cpu().numpy())

        predictions = np.concatenate(all_preds, axis=0)  # [N, 7]

        # 4. 构建返回 DataFrame (多列)
        pred_df = pd.DataFrame(
            predictions,
            columns=self.quantile_columns,  # ['&-q5', '&-q10', ..., '&-q95']
            index=unfiltered_df.index
        )

        # 5. 构建 do_predict 掩码 (检测异常输入)
        do_predict = np.ones(len(predictions), dtype=np.int_)
        # 可以添加异常检测逻辑...

        return pred_df, do_predict
```

### 6.5 自定义 Dataset

```python
class MultiScaleDataset(torch.utils.data.Dataset):
    """多时间尺度数据集"""

    def __init__(
        self,
        timestamps: np.ndarray,
        targets: Optional[np.ndarray],
        data_loader: MultiScaleDataLoader,
        timeframe_configs: dict,
    ):
        self.timestamps = pd.to_datetime(timestamps)
        self.targets = targets
        self.data_loader = data_loader
        self.timeframe_configs = timeframe_configs

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, idx):
        ts = self.timestamps[idx]

        # 从缓存加载多尺度窗口
        x_dict = self.data_loader.get_window(ts, self.timeframe_configs)

        # 转换为 tensor
        x_dict = {
            tf: torch.tensor(arr, dtype=torch.float32)
            for tf, arr in x_dict.items()
        }

        if self.targets is not None:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return x_dict, y
        else:
            return x_dict
```

### 6.6 配置文件示例

```json
{
    "freqai": {
        "enabled": true,
        "identifier": "multi_scale_quantile_v2",
        "train_period_days": 360,
        "backtest_period_days": 30,
        "live_retrain_hours": 24,

        "feature_parameters": {
            "include_timeframes": ["5m"],
            "label_period_candles": 48,
            "DI_threshold": 0
        },

        "model_training_parameters": {
            "quantiles": [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],

            "timeframe_configs": {
                "1d":  {"seq_len": 16,  "patch_size": 1},
                "4h":  {"seq_len": 48,  "patch_size": 1},
                "1h":  {"seq_len": 96,  "patch_size": 4},
                "15m": {"seq_len": 96,  "patch_size": 4},
                "5m":  {"seq_len": 144, "patch_size": 6}
            },

            "model_kwargs": {
                "d_model": 128,
                "nhead": 8,
                "num_encoder_layers": 2,
                "num_cross_scale_layers": 2,
                "dropout": 0.1
            },

            "n_epochs": 30,
            "batch_size": 32,
            "learning_rate": 1e-4
        }
    }
}
```

---

## 7. 训练策略

### 7.1 数据划分 (滚动窗口)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    历史数据时间轴 (2018-2025)                       │
│                                                                     │
│  ████████████████████████████████████████████████  │███│██│         │
│         训练集 (360 天)                               │  │          │
│                                                       │  │          │
│                        验证集 (30 天)                 │  │          │
│                                                          │          │
│                             测试集 (30 天)               │          │
└─────────────────────────────────────────────────────────────────────┘

滚动策略:
- 每 30 天向前滑动一次窗口
- 7 年数据约 70 次滚动测试
- 确保覆盖牛市、熊市、震荡市
```

### 7.2 训练超参数

```python
training_config = {
    'learning_rate': 1e-4,
    'batch_size': 32,
    'n_epochs': 30,
    'optimizer': 'AdamW',
    'weight_decay': 0.01,
    'lr_scheduler': 'CosineAnnealingWarmRestarts',
    'warmup_epochs': 3,
    'gradient_clip': 1.0,
    'early_stopping_patience': 5,
}
```

### 7.3 计算资源估算

```
单次训练:
- 训练样本: 360 天 × 288 (5m/天) 约 103,000 条
- 每 epoch: 103,000 / 32 约 3,200 个 batch
- 30 epochs: 约 96,000 次前向/反向传播

GPU 内存估算:
- 模型参数: 约 5M × 4 bytes = 20 MB
- 梯度: 20 MB
- 激活值 (batch=32, 136 tokens, d=128): 约 2 MB
- 总计: 约 50 MB (适合任何 GPU)

训练时间估算 (M1 Pro):
- 单 epoch: 约 2-3 分钟
- 完整训练: 约 1-2 小时
```

---

## 8. 验证指标

### 8.1 分位数校准度 (最重要)

```python
def calibration_score(y_true, y_pred_quantiles, quantiles):
    """
    计算每个分位数的校准度
    理想: coverage(q) 约等于 q
    """
    scores = {}
    for i, q in enumerate(quantiles):
        coverage = (y_true < y_pred_quantiles[:, i]).mean()
        scores[f'coverage_q{int(q*100)}'] = coverage
        scores[f'calibration_error_q{int(q*100)}'] = abs(coverage - q)
    return scores
```

### 8.2 区间覆盖率

```python
def interval_coverage(y_true, y_pred_q10, y_pred_q90):
    """80% 预测区间的实际覆盖率"""
    return ((y_true >= y_pred_q10) & (y_true <= y_pred_q90)).mean()
```

### 8.3 Pinball Loss

在测试集上计算平均 Pinball Loss。

---

## 9. 风险与挑战

### 9.1 技术风险

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 过拟合 | 回测好但实盘差 | 正则化、早停、滑动窗口验证 |
| 分位数交叉 | P10 > P50 (无效) | 单调性正则化损失 |
| 梯度爆炸 | 训练不稳定 | 梯度裁剪、Pre-LN |
| 数据加载慢 | 训练瓶颈 | 内存映射、预加载缓存 |

### 9.2 金融风险

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 数据偏移 | 市场状态变化 | 定期重训练 (FreqAI 支持) |
| 黑天鹅事件 | 尾部分位数失效 | 监控 P5/P95、仓位限制 |
| 过度自信 | 分位数区间太窄 | 监控校准度、宽度惩罚 |

---

## 10. 实现路线图

### Phase 1: 数据基础设施 (1-2 天)
1. 实现 `MultiScaleDataLoader` 类
2. 实现 `MultiScaleDataset` 类
3. 单元测试数据加载性能

### Phase 2: 模型核心 (2-3 天)
1. 实现 `TimeframeEncoder` (带 patch 压缩)
2. 实现 `CrossScaleAttention`
3. 实现 `MultiScaleQuantileTransformerModel`
4. 实现 `QuantileLoss`

### Phase 3: FreqAI 集成 (1-2 天)
1. 实现 `MultiScaleQuantileTransformer` (继承 BasePyTorchModel)
2. 重写 `fit()` 和 `predict()`
3. 测试与 FreqAI 的集成

### Phase 4: 策略与回测 (1-2 天)
1. 实现 `MultiScaleQuantileStrategy`
2. 配置文件编写
3. 回测验证

### Phase 5: 优化与部署 (持续)
1. 校准度调优
2. 性能优化
3. 文档完善

---

## 附录 A: 文件结构

```
user_data/
├── freqaimodels/
│   ├── __init__.py
│   ├── MultiScaleQuantileTransformer.py   # FreqAI 模型入口
│   └── torch/
│       ├── __init__.py
│       ├── model.py                        # PyTorch 网络
│       ├── data_loader.py                  # 数据加载器
│       ├── dataset.py                      # Dataset 类
│       └── loss.py                         # 损失函数
├── strategies/
│   └── MultiScaleQuantileStrategy.py
└── docs/
    └── multi_scale_quantile_model_design.md
```

---

## 附录 B: 参考资料

1. **Quantile Regression**: Koenker & Bassett (1978)
2. **Transformer**: Vaswani et al. "Attention is All You Need" (2017)
3. **Temporal Fusion Transformer**: Lim et al. (2019)
4. **Vision Transformer Patch**: Dosovitskiy et al. "An Image is Worth 16x16 Words" (2020)
5. **FreqAI 文档**: https://www.freqtrade.io/en/stable/freqai/

---

## TODO（待解决问题）

1. **模型属性未定义**: `MultiScaleQuantileTransformer` 直接访问 `self.pair`/`self.timeframe_configs`，但 `BasePyTorchModel` 没有这些属性。需要在 `__init__` 或 `fit()` 中显式设置，或调整数据加载器接口。
2. **时间戳来源错误**: `train_dataset = MultiScaleDataset(...)` 里使用 `data_dictionary["train_features"]['date']`，而 `filter_features()` 已移除 `date` 列，应改用 `data_dictionary["train_dates"]` 或预保存的原始时间戳。
3. **返回对象与推理路径不一致**: `fit()` 返回裸 `nn.Module`，`predict()` 却调用 `self.model.model.set_mode_for_inference()`。需要返回包含 `.model` 属性的封装类，或在推理时直接调用 `self.model` 并实现对应的 `eval()`/`set_mode_for_inference()`。
