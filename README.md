# Multivariate Gaussian Process Regression
A method for fast hyper-parameter optimization for multivariate Gaussian process regression

The method is written for easy integration into the sklearn framework. After initialization, the usual functions (`fit` and `predict`) are available.

The most interesting parameter is the `param_grid`. It should contain regularization parameters in `param_grid['alpha']` and Gaussian kernel parameters in `param_grid['gamma']`. Optionally, it can contain parameters for the integral operator-valued kernel linking functions in `param_grid['v']`.
