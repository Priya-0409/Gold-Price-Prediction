import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def get_gold_data():
    """Fetch historical gold price data from Yahoo Finance"""
    try:
        # Gold futures (GC=F) from Yahoo Finance
        data = yf.download('GC=F', start='2010-01-01', end=datetime.today().strftime('%Y-%m-%d'))
        return data[['Close']].rename(columns={'Close': 'Price'})
    except Exception as e:
        print(f"Error fetching gold data: {e}")
        return None

def prepare_data(data):
    """Prepare data for machine learning"""
    data = data.copy()
    data['Date'] = data.index
    data['Days'] = (data['Date'] - data['Date'].min()).dt.days
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    
    # Create features and target
    X = data[['Days', 'Year', 'Month', 'Day']]
    y = data['Price']
    
    return X, y

def train_model(X, y):
    """Train Random Forest regression model"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Model Mean Absolute Error: ${mae:.2f}")
    
    return model

def predict_future(model, last_date, days_to_predict=30):
    """Predict future gold prices"""
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict+1)]
    future_data = pd.DataFrame({
        'Date': future_dates,
        'Days': [(date - last_date).days + (last_date - pd.to_datetime('2010-01-01')).days for date in future_dates],
        'Year': [date.year for date in future_dates],
        'Month': [date.month for date in future_dates],
        'Day': [date.day for date in future_dates]
    })
    
    future_prices = model.predict(future_data[['Days', 'Year', 'Month', 'Day']])
    future_data['Predicted_Price'] = future_prices
    
    return future_data

def plot_results(historical, predictions):
    """Visualize historical and predicted prices"""
    plt.figure(figsize=(12, 6))
    plt.plot(historical.index, historical['Price'], label='Historical Prices', color='blue')
    plt.plot(predictions['Date'], predictions['Predicted_Price'], label='Predicted Prices', color='red', linestyle='--')
    plt.title('Gold Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Step 1: Get historical data
    gold_data = get_gold_data()
    if gold_data is None:
        return
    
    # Step 2: Prepare data
    X, y = prepare_data(gold_data)
    
    # Step 3: Train model
    model = train_model(X, y)
    
    # Step 4: Make predictions
    last_date = gold_data.index[-1]
    predictions = predict_future(model, last_date, days_to_predict=30)
    
    # Step 5: Display results
    print("\nNext 30 Days Gold Price Predictions:")
    print(predictions[['Date', 'Predicted_Price']].head(10))
    print("\nPrediction Summary:")
    print(f"Current Price: ${gold_data['Price'].iloc[-1]:.2f}")
    print(f"Predicted Price in 30 Days: ${predictions['Predicted_Price'].iloc[-1]:.2f}")
    print(f"Change: {(predictions['Predicted_Price'].iloc[-1] - gold_data['Price'].iloc[-1])/gold_data['Price'].iloc[-1]*100:.2f}%")
    
    # Step 6: Plot results
    plot_results(gold_data, predictions)

if __name__ == "__main__":
    main()