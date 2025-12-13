# Freqtrade 策略开发指南

## 项目概述

这是 freqtrade 的个人 fork，用于量化策略开发。基于稳定版 2025.11，不修改核心代码。

## 环境

```bash
source .venv/bin/activate
```

环境已配置完成，无需重复探索。

## 目录结构

```
user_data/
├── strategies/     # 策略代码 ✅ Git追踪
├── scripts/        # 辅助脚本 ✅ Git追踪
├── notebooks/      # 分析笔记本
├── data/           # K线数据 ❌ 不追踪
├── backtest_results/  # 回测结果 ❌ 不追踪
└── config*.json    # 配置文件 ❌ 不追踪(含密钥)
```

## 常用命令

```bash
# 下载数据
freqtrade download-data -p BTC/USDT -t 1h --exchange okx --timerange 20230101-

# 回测
freqtrade backtesting -s SampleStrategy --timerange 20230101-20231231

# 超参优化
freqtrade hyperopt -s SampleStrategy --hyperopt-loss SharpeHyperOptLoss -e 100

# 列出策略
freqtrade list-strategies

# 列出已下载数据
freqtrade list-data --exchange okx
```

## Git 工作流

```bash
# 远程仓库配置
origin   -> https://github.com/SubtleSpark/freqtrade.git  # 你的fork
upstream -> https://github.com/freqtrade/freqtrade.git    # 官方仓库

# 同步官方更新
git fetch upstream --tags
git merge 2025.12  # 合并新版本
git push
```

## 开发约定

- 策略文件使用 PascalCase 命名，如 `MyStrategy.py`
- 提交前会自动运行 pre-commit hooks (flake8, mypy, ruff)
- 策略代码放在 `user_data/strategies/`
- Git 提交信息规范：见 `.github/git-commit-instructions.md`
  - 使用中文、Conventional Commits 格式
  - 示例：`feat: 添加双均线策略`
