#preview the first 10 values of only the order placed at column
df[['Order Placed At']].head(10)


##example output from Copilot
import pandas as pd

# Example data
data = {'timestamp': ['11:38 PM, September 10 2024', '07:15 AM, July 19 2025']}
df = pd.DataFrame(data)

# Convert the timestamp column
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%I:%M %p, %B %d %Y')

# Now your timestamp column is in standard datetime format
print(df)

#convert the timestamp column
df['timestamp'] = pd.to_datetime(df['Order Placed At'], format='%I:%M %p, %B %d %Y')

#now your timestamp column is in standard datetime format
df[['timestamp']].head(10)
