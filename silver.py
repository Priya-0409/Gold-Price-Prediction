import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def get_silver_data():
    """Fetch historical silver price data from Yahoo Finance"""
    try:
        # Silver futures (SI=F) from Yahoo Finance
        data = yf.download('SI=F', start='2010-01-01', end=datetime.today().strftime('%Y-%m-%d'))
        return data[['Close']].rename(columns={'Close': 'Price'})
    except Exception as e:
        print(f"Error fetching silver data: {e}")
        return None

def prepare_data(data):
    """Prepare data for machine learning"""
    data = data.copy()
    data['Date'] = data.index
    data['Days'] = (data['Date'] - data['Date'].min()).dt.days
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['Weekday'] = data['Date'].dt.weekday  # Adding weekday as a feature
    
    # Create features and target
    X = data[['Days', 'Year', 'Month', 'Day', 'Weekday']]
    y = data['Price']
    
    # Add moving averages as features
    data['MA_7'] = data['Price'].rolling(window=7).mean()
    data['MA_30'] = data['Price'].rolling(window=30).mean()
    X = X.join(data[['MA_7', 'MA_30']].fillna(method='bfill'))
    
    return X, y

def train_model(X, y):
    """Train Random Forest regression model with improved parameters"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Model Mean Absolute Error: ${mae:.4f} per troy ounce")
    
    return model

def predict_future(model, last_date, days_to_predict=30):
    """Predict future silver prices with enhanced date features"""
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict+1)]
    
    # Create a temporary DataFrame to calculate moving averages
    temp_df = pd.DataFrame({'Date': future_dates, 'Price': [np.nan]*len(future_dates)})
    temp_df['MA_7'] = temp_df['Price'].rolling(window=7).mean().fillna(method='bfill')
    temp_df['MA_30'] = temp_df['Price'].rolling(window=30).mean().fillna(method='bfill')
    
    future_data = pd.DataFrame({
        'Date': future_dates,
        'Days': [(date - last_date).days + (last_date - pd.to_datetime('2010-01-01')).days for date in future_dates],
        'Year': [date.year for date in future_dates],
        'Month': [date.month for date in future_dates],
        'Day': [date.day for date in future_dates],
        'Weekday': [date.weekday() for date in future_dates],
        'MA_7': temp_df['MA_7'],
        'MA_30': temp_df['MA_30']
    })
    
    future_prices = model.predict(future_data[['Days', 'Year', 'Month', 'Day', 'Weekday', 'MA_7', 'MA_30']])
    future_data['Predicted_Price'] = future_prices
    
    return future_data

def plot_results(historical, predictions):
    """Enhanced visualization with more details"""
    plt.figure(figsize=(14, 7))
    
    # Historical prices
    plt.plot(historical.index, historical['Price'], 
             label='Historical Prices', 
             color='blue', 
             linewidth=2)
    
    # Predicted prices
    plt.plot(predictions['Date'], 
             predictions['Predicted_Price'], 
             label='Predicted Prices', 
             color='green', 
             linestyle='--', 
             linewidth=2)
    
    # Highlight current price
    current_price = historical['Price'].iloc[-1]
    plt.axhline(y=current_price, 
                color='red', 
                linestyle=':', 
                label=f'Current Price (${current_price:.2f})')
    
    plt.title('Silver Price Prediction (Troy Ounce)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Add annotations
    last_pred = predictions.iloc[-1]
    plt.annotate(f'Predicted: ${last_pred["Predicted_Price"]:.2f}\n{last_pred["Date"].strftime("%b %d")}', 
                 xy=(last_pred["Date"], last_pred["Predicted_Price"]),
                 xytext=(10, 10), 
                 textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                 arrowprops=dict(arrowstyle='->'))
    
    plt.tight_layout()
    plt.show()

def main():
    print("Fetching silver price data...")
    silver_data = get_silver_data()
    if silver_data is None:
        print("Failed to get silver data. Please check your internet connection.")
        return
    
    print("Preparing data for modeling...")
    X, y = prepare_data(silver_data)
    
    print("Training prediction model...")
    model = train_model(X, y)
    
    print("Making future predictions...")
    last_date = silver_data.index[-1]
    predictions = predict_future(model, last_date, days_to_predict=30)
    
    # Display results
    print("\nNext 30 Days Silver Price Predictions:")
    print(predictions[['Date', 'Predicted_Price']].head(10).to_string(index=False))
    
    print("\nPrediction Summary:")
    current_price = silver_data['Price'].iloc[-1]
    predicted_price = predictions['Predicted_Price'].iloc[-1]
    change_percent = (predicted_price - current_price)/current_price*100
    print(f"Current Price: ${current_price:.4f} per troy ounce")
    print(f"Predicted Price in 30 Days: ${predicted_price:.4f}")
    print(f"Change: {change_percent:.2f}%")
    
    print("\nGenerating visualization...")
    plot_results(silver_data, predictions)

if __name__ == "__main__":
    main()