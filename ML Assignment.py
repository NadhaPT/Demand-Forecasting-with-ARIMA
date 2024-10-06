#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
# Load the data
df1 = pd.read_csv(r"C:\Users\nadha.NICHU\OneDrive\Desktop\Transactional_data_retail_01.csv")
df2 = pd.read_csv(r"C:\Users\nadha.NICHU\OneDrive\Desktop\Transactional_data_retail_02.csv")
customer_data = pd.read_csv(r"C:\Users\nadha.NICHU\OneDrive\Desktop\CustomerDemographics.csv")
product_data = pd.read_csv(r"C:\Users\nadha.NICHU\OneDrive\Desktop\ProductInfo.csv")
df = pd.concat([df1, df2])


# In[26]:


top_products = df.groupby('StockCode').agg({'Quantity': 'sum'}).reset_index()
top_products = top_products.sort_values(by='Quantity', ascending=False).head(10)


# In[28]:


print(df.columns)


# In[3]:


print(df.head())


# In[5]:


import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load the data
df1 = pd.read_csv(r"C:\Users\nadha.NICHU\OneDrive\Desktop\Transactional_data_retail_01.csv")
df2 = pd.read_csv(r"C:\Users\nadha.NICHU\OneDrive\Desktop\Transactional_data_retail_02.csv")
customer_data = pd.read_csv(r"C:\Users\nadha.NICHU\OneDrive\Desktop\CustomerDemographics.csv")
product_data = pd.read_csv(r"C:\Users\nadha.NICHU\OneDrive\Desktop\ProductInfo.csv")

# Combine transactional data
df = pd.concat([df1, df2], ignore_index=True)

# Ensure 'InvoiceDate' is in a proper datetime format
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d %B %Y', errors='coerce')

# Summarizing top products by sales quantity
top_products = df.groupby('StockCode').agg({'Quantity': 'sum'}).reset_index()
top_products = top_products.sort_values(by='Quantity', ascending=False).head(10)

# Group data by weeks
df['Week'] = df['InvoiceDate'].dt.to_period('W').astype(str)
weekly_sales = df.groupby(['Week', 'StockCode']).agg({'Quantity': 'sum'}).reset_index()

# Choose a product from top_products
product_id = top_products['StockCode'].iloc[0]

# Filter weekly sales for the selected product
product_sales = weekly_sales[weekly_sales['StockCode'] == product_id]

# Ensure product_sales is not empty before proceeding
if not product_sales.empty:
    # ARIMA model (p=5, d=1, q=0)
    model = ARIMA(product_sales['Quantity'], order=(5, 1, 0))
    model_fit = model.fit()

    # Forecast next 15 weeks
    forecast = model_fit.forecast(steps=15)

    # Plot actual and forecasted sales
    plt.plot(product_sales['Quantity'], label='Actual Sales')
    plt.plot(range(len(product_sales), len(product_sales) + 15), forecast, label='Forecast')
    plt.legend()
    plt.title(f"Sales Forecast for StockCode: {product_id}")
    plt.xlabel('Weeks')
    plt.ylabel('Quantity Sold')
    plt.show()
else:
    print(f"No sales data available for product {product_id}.")


# In[6]:


import matplotlib.pyplot as plt

# Identify top 10 products
top_products = df.groupby('StockCode')['Quantity'].sum().sort_values(ascending=False).head(10)
top_products.plot(kind='bar')
plt.title('Top 10 Products by Quantity Sold')
plt.xlabel('Stock Code')
plt.ylabel('Total Quantity')
plt.show()


# In[9]:


pip install streamlit


# In[14]:


import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import pandas as pd

# Streamlit app layout
st.title('Retail Demand Forecasting')

# Load your data (assuming df is already loaded)
stock_code = st.sidebar.selectbox('Select Product', df['StockCode'].unique())
product_sales = df[df['StockCode'] == stock_code]

# Group by week using 'InvoiceDate' (since 'TransactionDate' does not exist)
weekly_sales = product_sales.groupby(pd.to_datetime(product_sales['InvoiceDate']).dt.to_period('W')).agg({'Quantity': 'sum'})

# Convert the Period index to datetime format for plotting
weekly_sales.index = weekly_sales.index.to_timestamp()

# Forecasting
model = ARIMA(weekly_sales['Quantity'], order=(5, 1, 0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=15)

# Display
st.write(f'Demand Forecasting for {stock_code}')
plt.figure(figsize=(10, 6))
plt.plot(weekly_sales.index, weekly_sales['Quantity'], label='Actual')
plt.plot(pd.date_range(weekly_sales.index[-1], periods=15, freq='W'), forecast, label='Forecast')
plt.legend()
st.pyplot(plt)


# In[1]:


get_ipython().system('pip install pandas matplotlib seaborn')


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    'Date': pd.date_range(start='2020-01-01', periods=100, freq='M'),
    'Actual Demand': pd.Series(range(100, 200)) + pd.Series([5 * x for x in range(100)]),
    'Predicted Demand': pd.Series(range(95, 195)) + pd.Series([4.5 * x for x in range(100)]),
}
df = pd.DataFrame(data)

# Plot Actual vs Predicted Demand
plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['Actual Demand'], label='Train Actual Demand', marker='o')
plt.plot(df['Date'], df['Predicted Demand'], label='Train Predicted Demand', marker='x')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.title(f"Actual vs Predicted Demand")
plt.legend()
plt.show()

# Error Distribution
train_error = df['Actual Demand'] - df['Predicted Demand']

# Create a figure for error distributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Training Error Distribution
sns.histplot(train_error, bins=10, kde=True, color='green', ax=ax1)
ax1.set_title('Training Error Distribution')
ax1.set_xlabel('Error')

# Simulate testing error (for demo purposes)
test_error = train_error + pd.Series([10 * x for x in range(100)])
sns.histplot(test_error, bins=10, kde=True, color='red', ax=ax2)
ax2.set_title('Testing Error Distribution')
ax2.set_xlabel('Error')

plt.show()


# In[ ]:




