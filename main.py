import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import time

# Replace 'your_file.csv' with the path to your CSV file
data = pd.read_csv("test.csv")


def lossFunction(m, b, points: pd.DataFrame):
    totatError = 0
    for i in range(len(points)):
        x = points.iloc[i].x
        y = points.iloc[i].y
        totatError += (y - (m * x + b)) ** 2
    return totatError / float(len(points))


def gradientDescent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    n = len(points)

    for i in range(n):
        x = points.iloc[i].x
        y = points.iloc[i].y

        m_gradient += -(2 / n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2 / n) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    loss = lossFunction(m, b, points)
    return m, b, loss


m = 0
b = 0
L = 0.0005
epochs = 1000

with open("0.0003.csv", "w") as f:
    start = time.time()
    for i in range(epochs):
        m, b, loss = gradientDescent(m, b, data, L)
        print(m, b, loss, sep=",", file=f)
    end = time.time()

print(m, b)
print(f"Time: {end - start}")
plt.scatter(data.x, data.y, c="black", marker=".")
plt.plot(
    list(range(int(min(data.x)), int(max(data.y)))),
    [m * x + b for x in range(int(min(data.x)), int(max(data.y)))],
    color="red",
)
plt.show()
