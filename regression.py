import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime

# Fetches historical stock data from Yahoo Finance
def fetch_data(stock, start_date, end_date):
    data = yf.download(stock, start=start_date, end=end_date)
    data.reset_index(inplace=True)  # Reset index to make the date a column
    data['Symbol'] = stock  # Add a column to denote the stock symbol
    return data

# Prepares data for regression analysis
def prepare_data(data):
    data['DateOrdinal'] = data['Date'].apply(lambda x: x.toordinal())  # Convert dates to ordinal for regression
    return data[['DateOrdinal', 'Close']]  # Focus on the date and closing price

# Performs linear regression
def linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)  # Fit the linear regression model to the data
    return model

# Performs polynomial regression
def polynomial_regression(X, y, degree):
    poly_features = PolynomialFeatures(degree=degree)  # Create polynomial features
    X_poly = poly_features.fit_transform(X)  # Transform the data to polynomial features
    model = LinearRegression()
    model.fit(X_poly, y)  # Fit the linear regression model to the polynomial-transformed data
    return model, poly_features

# Performs logarithmic regression
def logarithmic_regression(X, y):
    X_transformed = np.log(X)  # Apply logarithmic transformation to the data
    model = LinearRegression()
    model.fit(X_transformed, y)  # Fit the linear regression model to the logarithmically-transformed data
    return model

# Plots combined regression results for a given stock
def plot_combined_regression(data, models, model_names, poly_features=None):
    plt.figure(figsize=(12, 8))
    plt.scatter(data['Date'], data['Close'], color='blue', label='Actual Prices')
    colors = ['red', 'green', 'purple']  # Different colors for each regression model

    for model, name, color in zip(models, model_names, colors):
        X = data['DateOrdinal'].values.reshape(-1, 1)
        if name == 'Polynomial':
            X = poly_features.transform(X)  # Use polynomial features for polynomial regression
        elif name == 'Logarithmic':
            X = np.log(X)  # Use logarithmic transformation for logarithmic regression
        y_pred = model.predict(X)
        plt.plot(data['Date'], y_pred, label=f'{name} Regression Line', color=color)
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title(f'Regression Analysis for {data["Symbol"].iloc[0]}')
    plt.legend()
    plt.show()

# Store and print MSE and R2 results for each model
mse_values = {}
r2_values = {}

def store_and_print_results(model_name, stock, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mse_values[f'{stock}_{model_name}'] = mse
    r2_values[f'{stock}_{model_name}'] = r2
    print(f'{model_name} MSE for {stock}: {mse}')
    print(f'{model_name} R2 for {stock}: {r2}')

# Main execution block
if __name__ == "__main__":
    stocks = ['GOOGL', 'AAPL']  # Stocks to analyze
    start_date = '2018-01-01'  # Start date for data
    end_date = datetime.now().strftime('%Y-%m-%d')  # End date for data (current date)

    for stock in stocks:
        try:
            data = fetch_data(stock, start_date, end_date)
            prepared_data = prepare_data(data)
            X = prepared_data['DateOrdinal'].values.reshape(-1, 1)
            y = prepared_data['Close'].values

            linear_model = linear_regression(X, y)
            poly_model, poly_features = polynomial_regression(X, y, degree=2)
            log_model = logarithmic_regression(X, y)

            store_and_print_results('Linear Regression', stock, y, linear_model.predict(X))
            store_and_print_results('Polynomial Regression', stock, y, poly_model.predict(poly_features.transform(X)))
            store_and_print_results('Logarithmic Regression', stock, y, log_model.predict(np.log(X)))

            print(f'Plotting combined regression analysis for {stock}')
            plot_combined_regression(
                data, 
                [linear_model, poly_model, log_model], 
                ['Linear', 'Polynomial', 'Logarithmic'], 
                poly_features=poly_features
            )

        except Exception as e:
            print(f"Error processing {stock}: {e}")

    print('All processing complete.')

# Plot comparison charts for MSE and R2 values
def plot_comparison_charts(mse_values, r2_values):
    plt.figure(figsize=(10, 5))
    plt.bar(mse_values.keys(), mse_values.values(), color=['blue', 'blue', 'blue', 'red', 'red', 'red'])
    plt.title('MSE Comparison for Regression Models')
    plt.ylabel('MSE')
    plt.xticks(rotation=45)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.bar(r2_values.keys(), r2_values.values(), color=['blue', 'blue', 'blue', 'red', 'red', 'red'])
    plt.title('R² Comparison for Regression Models')
    plt.ylabel('R² Score')
    plt.xticks(rotation=45)
    plt.show()

plot_comparison_charts(mse_values, r2_values)
