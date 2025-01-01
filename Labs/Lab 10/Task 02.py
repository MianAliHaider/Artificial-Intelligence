import numpy as np

def calculate_mean(values):
    return np.mean(values)

def calculate_slope(X, Y, mean_X, mean_Y):
    numerator = np.sum((X - mean_X) * (Y - mean_Y))
    denominator = np.sum((X - mean_X) ** 2)
    return numerator / denominator

def calculate_intercept(mean_X, mean_Y, slope):
    return mean_Y - (slope * mean_X)

def predict(X, theta_0, theta_1):
    return theta_0 + theta_1 * X

def calculate_mse(Y, Y_pred):
    return np.mean((Y - Y_pred) ** 2)

def fit_linear_regression(X, Y):
    mean_X = calculate_mean(X)
    mean_Y = calculate_mean(Y)
    
    slope = calculate_slope(X, Y, mean_X, mean_Y)
    intercept = calculate_intercept(mean_X, mean_Y, slope)
    
    Y_pred = predict(X, intercept, slope)
    mse = calculate_mse(Y, Y_pred)
    
    return slope, intercept, mse

X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 5, 7, 8])

slope, intercept, mse = fit_linear_regression(X, Y)
print(f"Slope: {slope},\nIntercept: {intercept}, \nMSE: {mse}")
