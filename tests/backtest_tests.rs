use quant_rs::backtest::engine::BacktestEngine;
use quant_rs::core::QuantError;
use quant_rs::strategy::signal::Signal;

fn assert_approx_eq(a: f64, b: f64) {
    let eps = 1e-12_f64.max(a.abs().max(b.abs()) * 1e-12);
    assert!(
        (a - b).abs() < eps,
        "expected {a} ≈ {b}, diff {}",
        (a - b).abs()
    );
}

// ── backtest_engine_run ───────────────────────────────────────────────────────

#[test]
fn run_hold_only_keeps_flat_equity() {
    let prices = [100.0, 110.0, 90.0, 120.0];
    let signals = [Signal::Hold, Signal::Hold, Signal::Hold, Signal::Hold];

    let result = BacktestEngine::run(&prices, &signals).unwrap();

    assert_eq!(result.strategy_returns.len(), 3);
    assert_approx_eq(result.strategy_returns[0], 0.0);
    assert_approx_eq(result.strategy_returns[1], 0.0);
    assert_approx_eq(result.strategy_returns[2], 0.0);

    assert_eq!(result.equity_curve.len(), 4);
    assert_approx_eq(result.equity_curve[0], 1.0);
    assert_approx_eq(result.equity_curve[1], 1.0);
    assert_approx_eq(result.equity_curve[2], 1.0);
    assert_approx_eq(result.equity_curve[3], 1.0);
    assert_approx_eq(result.final_equity, 1.0);
}

#[test]
fn run_buy_then_hold_captures_returns() {
    let prices = [100.0, 110.0, 121.0];
    let signals = [Signal::Buy, Signal::Hold, Signal::Hold];

    let result = BacktestEngine::run(&prices, &signals).unwrap();

    assert_eq!(result.strategy_returns.len(), 2);
    assert_approx_eq(result.strategy_returns[0], 0.1);
    assert_approx_eq(result.strategy_returns[1], 0.1);

    assert_eq!(result.equity_curve.len(), 3);
    assert_approx_eq(result.equity_curve[0], 1.0);
    assert_approx_eq(result.equity_curve[1], 1.1);
    assert_approx_eq(result.equity_curve[2], 1.21);
    assert_approx_eq(result.final_equity, 1.21);
}

#[test]
fn run_sell_before_period_stays_out_of_market() {
    let prices = [100.0, 110.0, 120.0];
    let signals = [Signal::Sell, Signal::Buy, Signal::Hold];

    let result = BacktestEngine::run(&prices, &signals).unwrap();

    assert_eq!(result.strategy_returns.len(), 2);
    assert_approx_eq(result.strategy_returns[0], 0.0);
    assert_approx_eq(result.strategy_returns[1], 120.0 / 110.0 - 1.0);
}

#[test]
fn run_signal_applies_to_next_period() {
    let prices = [100.0, 90.0, 99.0, 108.9];
    let signals = [Signal::Buy, Signal::Sell, Signal::Buy, Signal::Hold];

    let result = BacktestEngine::run(&prices, &signals).unwrap();

    // i=1 uses signals[0]=Buy   -> in position -> -10%
    // i=2 uses signals[1]=Sell  -> out         -> 0%
    // i=3 uses signals[2]=Buy   -> in position -> +10%
    assert_eq!(result.strategy_returns.len(), 3);
    assert_approx_eq(result.strategy_returns[0], -0.1);
    assert_approx_eq(result.strategy_returns[1], 0.0);
    assert_approx_eq(result.strategy_returns[2], 0.1);

    assert_eq!(result.equity_curve.len(), 4);
    assert_approx_eq(result.equity_curve[0], 1.0);
    assert_approx_eq(result.equity_curve[1], 0.9);
    assert_approx_eq(result.equity_curve[2], 0.9);
    assert_approx_eq(result.equity_curve[3], 0.99);
    assert_approx_eq(result.final_equity, 0.99);
}

#[test]
fn run_lengths_are_consistent() {
    let prices = [100.0, 105.0, 110.0, 100.0, 95.0];
    let signals = [
        Signal::Buy,
        Signal::Hold,
        Signal::Hold,
        Signal::Sell,
        Signal::Hold,
    ];

    let result = BacktestEngine::run(&prices, &signals).unwrap();
    assert_eq!(result.strategy_returns.len(), prices.len() - 1);
    assert_eq!(result.equity_curve.len(), prices.len());
}

#[test]
fn run_compounds_equity_from_strategy_returns() {
    let prices = [100.0, 110.0, 99.0, 108.9];
    let signals = [Signal::Buy, Signal::Hold, Signal::Hold, Signal::Hold];

    let result = BacktestEngine::run(&prices, &signals).unwrap();
    let compounded = result
        .strategy_returns
        .iter()
        .fold(1.0, |acc, r| acc * (1.0 + r));
    assert_approx_eq(result.final_equity, compounded);
}

#[test]
fn run_insufficient_data_is_error() {
    let signals = [Signal::Hold];
    assert!(matches!(
        BacktestEngine::run(&[100.0], &signals),
        Err(QuantError::InsufficientData)
    ));
    assert!(matches!(
        BacktestEngine::run(&[], &[]),
        Err(QuantError::InsufficientData)
    ));
}

#[test]
fn run_signals_length_mismatch_is_error() {
    let prices = [100.0, 110.0, 120.0];
    let signals = [Signal::Buy, Signal::Hold];
    assert!(matches!(
        BacktestEngine::run(&prices, &signals),
        Err(QuantError::InvalidValue(_))
    ));
}

#[test]
fn run_non_positive_price_is_error() {
    let signals = [Signal::Hold, Signal::Hold];
    assert!(matches!(
        BacktestEngine::run(&[100.0, 0.0], &signals),
        Err(QuantError::NonPositivePrice(_))
    ));
    assert!(matches!(
        BacktestEngine::run(&[100.0, -1.0], &signals),
        Err(QuantError::NonPositivePrice(_))
    ));
}

#[test]
fn run_invalid_price_is_error() {
    let signals = [Signal::Hold, Signal::Hold];
    assert!(matches!(
        BacktestEngine::run(&[100.0, f64::NAN], &signals),
        Err(QuantError::InvalidValue(_))
    ));
    assert!(matches!(
        BacktestEngine::run(&[100.0, f64::INFINITY], &signals),
        Err(QuantError::InvalidValue(_))
    ));
}
