import csv
import sys
csv.field_size_limit(sys.maxsize)
with open("C:\\Users\\yookw\\Downloads\\coding\\movies.csv", newline='', encoding="cp1252", errors="replace") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("C:\\Users\\yookw\\Downloads\\coding\\movies.csv", encoding="utf-8", on_bad_lines='skip')
df = df.dropna()
X = df[['runtime', 'popularity', 'vote_count']]
y = df['vote_average']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

print(model.intercept_)
print(model.coef_)

y_pred = model.predict(X_test)
predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(predictions.head())

r2 = r2_score(y_test, y_pred)
print(f"R_squared score: {r2}")