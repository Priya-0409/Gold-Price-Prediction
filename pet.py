import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load petrol price data (example CSV from Indian metro cities)
petrol_df = pd.read_csv('petrol_prices_india.csv', parse_dates=['date'])

# Aggregate petrol prices to get average price per day (or per month)
daily_petrol = petrol_df.groupby('date')['value'].mean().reset_index()
daily_petrol.rename(columns={'value': 'petrol_price'}, inplace=True)

# Load crude oil price data
crude_df = pd.read_csv('crude_oil_prices.csv', parse_dates=['Date'])
crude_df.rename(columns={'Date': 'date', 'Close': 'crude_price'}, inplace=True)

# Merge petrol and crude oil prices on date
data = pd.merge(daily_petrol, crude_df[['date', 'crude_price']], on='date', how='inner')

# Feature engineering: extract year and month
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month

# Define features and target
X = data[['year', 'month', 'crude_price']]
y = data['petrol_price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate model
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print(f'RMSE: {rmse:.4f}')
print(f'R² Score: {r2:.4f}')

# Plot actual vs predicted petrol prices
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel('Actual Petrol Price')
plt.ylabel('Predicted Petrol Price')
plt.title('Actual vs Predicted Petrol Prices')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.show()
