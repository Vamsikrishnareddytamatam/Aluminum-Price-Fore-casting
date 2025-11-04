import pandas as pd
import numpy as np
df = pd.read_csv("Aluminium Historical Data.csv")
df.head()
df.tail()
all_business_days = pd.date_range(start='2020-01-02', end=pd.to_datetime("today"), freq='B')
print(all_business_days)
df1 = df.iloc[:,[0,1]]
print(df1.head())
import pandas as pd

df2 = df1.copy()

def parse_dates(date):
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return pd.to_datetime(date, format=fmt)
        except (ValueError, TypeError):
            pass
    return pd.NaT

df1["Date"] = df1["Date"].apply(parse_dates)
merged_df1 = df1.copy()
start_date_row = merged_df1[merged_df1["Date"] == merged_df1["Date"].min()]
end_date_row = merged_df1[merged_df1["Date"] == merged_df1["Date"].max()]
# Generate full range of business days
start_date = merged_df1["Date"].min()
end_date = merged_df1["Date"].max()
full_business_days = pd.date_range(start=start_date, end=end_date, freq="B")  # 'B' means business days

# Reindex DataFrame to include missing business days
df_full = pd.DataFrame({"Date": full_business_days})
df_merged = df_full.merge(merged_df1, on="Date", how="left")  # Merge to keep all business days

df_merged["Is_Non_Business_Day"] = df_merged["Date"].dt.weekday >= 5  # Saturday=5, Sunday=6

# Filter only non-business day rows
non_business_days = df_merged[df_merged["Is_Non_Business_Day"]]

# Display the results
if not non_business_days.empty:
    print("Non-Business Days Found:")
    print(non_business_days[["Date"]])
else:
    print("All dates are business days!")


df_merged1 = df_merged.drop("Is_Non_Business_Day", axis=1)
df_merged1.info()
merged_df1 = merged_df1.ffill()
merged_df1.info()
merged_df1['Price'] = merged_df1['Price'].replace(',', '', regex=True).astype(float)
mean_price = merged_df1['Price'].mean()
median_price = merged_df1['Price'].median()
mode_price = merged_df1['Price'].mode()

print("Mean Price:", mean_price)
print("Median Price:", median_price)
print("Mode price:", mode_price)
variance = merged_df1['Price'].var()
print("Variance:", variance)


std_dev = merged_df1['Price'].std()
print("Standard Deviation:", std_dev)


skewness = merged_df1['Price'].skew()
print("Skewness:", skewness)


kurtosis = merged_df1['Price'].kurtosis()
print("Kurtosis:", kurtosis)
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8,5))
sns.histplot(df['Price'], bins=30, kde=True, color='blue')  # KDE adds a smooth curve
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.title("Histogram of Aluminium Prices")
plt.show()
plt.figure(figsize=(6,4))
sns.boxplot(y=df['Price'], color='green')
plt.ylabel("Price")
plt.title("Boxplot of Aluminium Prices")
plt.show()
import pandas as pd

# Load dataset
df = pd.read_csv("Aluminium Historical Data.csv")

# Inspect the first few rows
print(df.head())

# Convert 'Date' column to datetime format, handling mixed formats
df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)

# Generate a complete range of business days
all_business_days = pd.date_range(start='2020-01-01', end=pd.to_datetime("today"), freq='B')

# Create a new DataFrame with all business days
full_df = pd.DataFrame({'Date': all_business_days})

# Merge with the original dataset
df_filled = full_df.merge(df, on='Date', how='left')

# Save the updated dataset
df_filled.to_csv("Aluminium_Historical_Data_Filled.csv", index=False)

print("Missing dates added, and file saved as 'Aluminium_Historical_Data_Filled.csv'")

df_filled.info()

# after outliers replace

dd = pd.read_csv("Aluminium_Historical_Data_Interpolated.csv")
dd
plt.figure(figsize=(6,4))
sns.boxplot(y=dd['Price'], color='green')
plt.ylabel("Price")
plt.title("Boxplot of Aluminium Prices")
plt.show()
plt.figure(figsize=(8,5))
sns.histplot(dd['Price'], bins=30, kde=True, color='blue')  # KDE adds a smooth curve
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.title("Histogram of Aluminium Prices")
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load dataset
file_path = "Aluminium_Historical_Data_Interpolated.csv"
df = pd.read_csv(file_path)

# Convert Date column to datetime and extract features
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.dayofweek

# Define features and target
X = df[["Year", "Month", "Day", "DayOfWeek"]]
y = df["Price"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Create a DataFrame to store actual vs predicted prices
predictions_df = pd.DataFrame({
    "Date": df["Date"].iloc[-len(y_test):].values,
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

# Display the first few predicted values
print(predictions_df.head())

# Generate future dates for prediction
future_dates = pd.date_range(start=df["Date"].max() + pd.Timedelta(days=1), periods=30)
future_features = pd.DataFrame({
    "Year": future_dates.year,
    "Month": future_dates.month,
    "Day": future_dates.day,
    "DayOfWeek": future_dates.dayofweek
})

# Predict future prices
future_prices = model.predict(future_features)

# Create DataFrame for future predictions
future_df = pd.DataFrame({"Date": future_dates, "Predicted Price": future_prices})
print("Future Price Predictions:")
print(future_df)

# Plot actual, predicted, and future prices
plt.figure(figsize=(12, 6))
plt.plot(df["Date"].iloc[-len(y_test):], y_test, label="Actual Price", color='blue')
plt.plot(df["Date"].iloc[-len(y_test):], y_pred, label="Predicted Price", color='red')
plt.plot(future_df["Date"], future_df["Predicted Price"], label="Future Price", color='green', linestyle='dashed')
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Actual, Predicted, and Future Aluminium Prices")
plt.legend()
plt.grid()
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Load the dataset
df = pd.read_csv("Aluminium_Historical_Data_Interpolated.csv")

# Convert Date column to datetime format and set as index
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

# Plot the time series data
plt.figure(figsize=(12, 5))
plt.plot(df["Price"], label="Aluminium Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Aluminium Price Over Time")
plt.legend()
plt.show()

# Ignore convergence warnings
warnings.filterwarnings("ignore")

# Fit an ARIMA model (p=2, d=1, q=2)
model = ARIMA(df["Price"], order=(2,1,2))
model_fit = model.fit()

# Print model summary
print(model_fit.summary())

# Forecast the next 30 days
forecast_steps = 30
forecast = model_fit.forecast(steps=forecast_steps)
forecast_dates = pd.date_range(df.index[-1], periods=forecast_steps+1, freq='B')[1:]
forecast_df = pd.DataFrame({"Date": forecast_dates, "Predicted Price": forecast.values})

# Save predicted prices to CSV
forecast_df.to_csv("Predicted_Aluminium_Prices.csv", index=False)

# Plot forecast
plt.figure(figsize=(12, 5))
plt.plot(df.index, df["Price"], label="Historical Prices")
plt.plot(forecast_df["Date"], forecast_df["Predicted Price"], label="Forecast", linestyle='dashed', color='red')
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Aluminium Price Forecast (Next 30 Days)")
plt.legend()
plt.show()

# Display predicted prices
print(forecast_df)
#XGBoost Regressor


import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
file_path = "Aluminium_Historical_Data_Interpolated.csv"
df = pd.read_csv(file_path)

# Convert Date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')  # Ensure chronological order

# Set features and target variable
df['Days_Since_Start'] = (df['Date'] - df['Date'].min()).dt.days  # Convert date to numerical values
features = ['Days_Since_Start']  # You can add more features if available
target = 'Price'  # Adjust column name if necessary

X = df[features]
y = df[target]

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost model
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, objective='reg:squarederror')
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"MAE: {mae}, RMSE: {rmse}")

# Future Prediction (Next 30 Days)
future_dates = pd.date_range(start=df['Date'].max(), periods=30, freq='D')  # Generate next 30 days
future_days_since_start = (future_dates - df['Date'].min()).days.to_numpy().reshape(-1, 1)  # Convert to numerical format
future_days_scaled = scaler.transform(future_days_since_start)  # Apply same scaling

future_prices = model.predict(future_days_scaled)

# Create a DataFrame for future predictions
future_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_prices})
print(future_df)

# Plot Results
plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['Price'], label="Actual Prices", color='blue')
plt.plot(future_dates, future_prices, label="Predicted Prices", color='red', linestyle="dashed")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=45)
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta

# Load data
df = pd.read_csv("Aluminium_Historical_Data_Interpolated.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)

# Extract day, month, and year as features
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Store the last available date before dropping
last_date = df['Date'].iloc[-1]

# Drop original Date column
df.drop(columns=['Date'], inplace=True)

# Prepare features and target
X = df[['Day', 'Month', 'Year']]
y = df['Price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

# Generate next 30 days
future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]  # Next 30 days

# Create dataframe for future dates
future_df = pd.DataFrame({
    'Day': [date.day for date in future_dates],
    'Month': [date.month for date in future_dates],
    'Year': [date.year for date in future_dates]
})

# Predict future prices
future_prices = model.predict(future_df)

# Plot actual vs predicted prices
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual Prices', color='blue')
plt.plot(y_pred, label='Predicted Prices', color='red', linestyle='dashed')
plt.axvline(x=len(y_test), color='black', linestyle='dotted', label="Prediction Start")
plt.plot(range(len(y_test), len(y_test) + 30), future_prices, label="Next 30 Days", color='green')

plt.xlabel("Samples")
plt.ylabel("Price")
plt.title("Actual vs Predicted Aluminium Prices with 30-Day Forecast")
plt.legend()
plt.show()

# Print next 30 days predicted prices
future_df['Predicted Price'] = future_prices
print(future_df)