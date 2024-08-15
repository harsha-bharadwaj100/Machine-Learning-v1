import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

# Replace 'your_file.csv' with the path to your CSV file
df = pd.read_csv("test.csv")

# Display the first few rows of the dataframe
# print(df.head())

data = df.to_dict()

x1 = df["x"].to_list()[:50]

y1 = list(map(int, df["y"].to_list()[:50]))

print(max(x1), max(y1))
plt.plot(x1, y1, "o", marker=".", markersize=2)


def squareLoss(x, y, func):
    yl = list(map(func, x))
    result = 0
    for i, j in zip(y, yl):
        result += (j - i) ** 2
    return result


global m, c
m, c = 1, 0.1


def yl(x):
    global m, c
    return m * x + c


print(f"Loss: {squareLoss(x1, y1, yl)}")

# a, b, d = sp.symbols('a b d')
# sp.solve()
# ans = sp.solve([sp.Eq(a*squareLoss(x1, y1, yl), Y1), sp.Eq(((.173*n)+(.517*n))/165, Y2), 
#       sp.Eq(Y1+Y2, 1)], [n, Y1, Y2])

x = np.linspace(0, 100, 1000)
y = x


# Create the plot
plt.plot(x, y)
# plt.plot()
plt.show()
# print(x)
