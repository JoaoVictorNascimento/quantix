pub fn simple_returns(prices: &[f64]) -> Vec<f64> {
    prices
        .windows(2)
        .map(|w| (w[1] / w[0]) - 1.0)
        .collect()
}

pub fn log_returns(prices: &[f64]) -> Vec<f64> {
    prices
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect()
}