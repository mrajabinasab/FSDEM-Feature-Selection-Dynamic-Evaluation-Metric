import numpy as np

def approx_func(x, y):
    def f(x_val):
        if np.max(x_val) > max(x):
            raise ValueError("x is larger than max observed value.")
        return np.interp(x_val, x, y)
    def df(x_val):
        if np.max(x_val) > max(x):
            raise ValueError("x is larger than max observed value.")
        h = 1e-7  
        return (f(x_val + h) - f(x_val - h)) / (2 * h)
    return f, df

def fsdem(f, start, end, n=1000):
    x_values = np.linspace(start, end, n)
    y_values = f(x_values)
    area = np.trapz(y_values, x_values)
    return area / (end-start)

def stability(dx, start, end):
    val = 0
    for i in range(end-start+1):
        val += dx(start + i)
    return val/(end-start+1)

'''
USAGE:
1. Use different observations of the metric you desire and its corresponding values for the number of features to approximate a function and its derivatives using the approx_func function.
2. Use the fsdem function to calculate the FSDEM score of the algorithm over the range between start and end.
3. Use the stability function to calculate the stability score of the algorithm over the range between start and end.

Example:
# Observations of the metric and corresponding number of features
x = [1, 2, 3, 4, 5]
y = [0.1, 0.4, 0.6, 0.8, 0.9]

# Approximate the function and its derivative
f, df = approx_func(x, y)

# Calculate FSDEM score
fsdem_score = fsdem(f, start=1, end=5)
print("FSDEM Score:", fsdem_score)

# Calculate stability score
stability_score = stability(df, start=1, end=5)
print("Stability Score:", stability_score)
'''