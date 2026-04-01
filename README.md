# quant-rs

A Rust library for quantitative finance, providing tools for financial metrics, portfolio analysis, trading strategies, and backtesting.

---

## 🚀 Overview

`quant-rs` is designed to help developers and researchers analyze financial time series and evaluate trading strategies.

It follows a modular pipeline:

prices → returns → metrics → strategy → backtest


Each layer builds on top of the previous one, enabling a clear and composable workflow.

---

## ✨ Features

### 📈 Returns
- Simple returns
- Log returns
- Cumulative returns

### 📊 Risk Metrics
- Variance
- Volatility
- Annualized volatility

### ⚖️ Performance Metrics
- Sharpe ratio
- Sortino ratio

### 📉 Drawdown Analysis
- Drawdown series
- Maximum drawdown
- Drawdown duration

### 📦 Portfolio
- Weighted portfolio returns
- Position-based modeling

### 🧠 Strategy
- Moving average crossover signals

### 🔄 Backtesting
- Signal-driven execution engine
- Equity curve tracking
- Strategy performance evaluation

---

## ⚡ Quick Example

```rust
use quant_rs::strategy::moving_average_crossover_signals;
use quant_rs::backtest::BacktestEngine;

fn main() {
    let prices = vec![100.0, 102.0, 105.0, 103.0, 110.0, 115.0];

    let signals = moving_average_crossover_signals(&prices, 2, 3).unwrap();

    let result = BacktestEngine::run(&prices, &signals).unwrap();

    println!("Final equity: {:.4}", result.final_equity);

    let sharpe = result.sharpe_ratio(0.0).unwrap();
    println!("Sharpe ratio: {:.4}", sharpe);
}
```
📦 Installation

Add to your Cargo.toml

[dependencies]
`quant-rs = { path = "." }`

🧱 Project Structure


```
src/
├─ metrics/     # financial calculations
├─ portfolio/   # asset aggregation
├─ strategy/    # trading signals
├─ backtest/    # simulation engine
└─ core/        # shared types and errors
```

🧠 Concepts

This library is built around key quantitative finance principles:

- Returns are the foundation of all financial metrics
- Volatility measures uncertainty and dispersion
- Sharpe ratio evaluates risk-adjusted performance
- Drawdown captures real-world losses
- Backtesting validates strategies using historical data

For more details, see:
```
docs/fundamentals.md
```

⚠️ Design Principles
- Functional core (metrics)
- Domain modeling (portfolio, strategy)
- Composable pipeline
- Explicit error handling (Result)

🤝 Contributing

Contributions are welcome.

If you find a bug or have an idea, feel free to open an issue or submit a pull request.
