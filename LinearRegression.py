# Simple Linear Regression on Birth Weight Data
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Load dataset
df = pd.read_csv("birthwt.csv")

# Extract variables
age = df["age"].to_numpy()
birthwt = df["bwt"].to_numpy()
motherwt = df["lwt"].to_numpy()

# Plot Age vs Birth Weight
plt.scatter(age, birthwt/1000)
plt.xlabel("Mother's Age")
plt.ylabel("Baby's Birth Weight (kg)")
plt.title("Age vs Birth Weight")
plt.show()

# Linear Regression: Age → Birth Weight
lr = LinearRegression()
age = age.reshape(-1, 1)  # reshape for sklearn
lr.fit(age, birthwt)
print("Age vs Birth Weight")
print("Coefficient:", lr.coef_[0])
print("Intercept:", lr.intercept_)
print()

# Plot Age vs Mother's Weight
plt.scatter(age, motherwt)
plt.xlabel("Mother's Age")
plt.ylabel("Mother's Weight (lbs)")
plt.title("Age vs Mother's Weight")
plt.show()

# Linear Regression: Age → Mother's Weight
lr.fit(age, motherwt)
print("Age vs Mother's Weight")
print("Coefficient:", lr.coef_[0])
print("Intercept:", lr.intercept_)
print()

# Linear Regression: Birth Weight → Mother's Weight
birthwt = birthwt.reshape(-1, 1)
lr.fit(birthwt, motherwt)
print("Birth Weight vs Mother's Weight")
print("Coefficient:", lr.coef_[0])
print("Intercept:", lr.intercept_)
print()
