import numpy as np

def add_bias_column(X):
    """Add an intercept column of ones to feature matrix X (m, n) -> (m, n+1)."""
    m = X.shape[0]
    return np.hstack([np.ones((m, 1)), X])

def train_linear_regression_normal(X, y):
    """
    Train linear regression via the normal equation.

    X : (m, n)  feature matrix (no bias column)
    y : (m,)    target vector (e.g., log1p(msrp))

    Returns:
        w0 : float           bias (intercept)
        w  : (n,) np.ndarray weights for the n features
    """
    X_aug = add_bias_column(X)                # (m, n+1)
    XtX   = X_aug.T @ X_aug                   # (n+1, n+1)
    Xty   = X_aug.T @ y                       # (n+1,)

    # Prefer solve() to inv() for numerical stability
    w_aug = np.linalg.solve(XtX, Xty)         # (n+1,)
    w0, w = w_aug[0], w_aug[1:]
    return w0, w

def train_regularized_linear_regression(X, y, r):
    """
    Train regularized linear regression via the normal equation.
    
    X : (m, n)  feature matrix (no bias column)
    y : (m,)    target vector
    r : float   regularization parameter
    
    Returns:
        w0 : float           bias (intercept)
        w  : (n,) np.ndarray weights for the n features
    """
    X_aug = add_bias_column(X)
    n = X_aug.shape[1]
    
    # Create identity matrix for regularization (we don't regularize bias)
    I = np.eye(n)
    I[0, 0] = 0  # Don't regularize the bias term
    
    # Normal equation with regularization: (X'X + rI)w = X'y
    XtX = X_aug.T @ X_aug
    Xty = X_aug.T @ y
    
    # Solve for w
    w_aug = np.linalg.solve(XtX + r * I, Xty)
    
    w0, w = w_aug[0], w_aug[1:]
    return w0, w

def predict_linear(X, w0, w):
    """Predict in the model's target space (here: log1p)."""
    return w0 + X @ w

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
