####Data Types####

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


####Data Visualization####
#preview data where the price is greater than 10000
df_cleaned[df_cleaned[‘Price’] > 10000]

#count the number of occurances for each value in the timestamp column
df_aggregated = pd.DataFrame(df.groupby('timestamp')['Order ID'].count()).reset_index()

#preview the results
df_aggregated.head()

#create new column of just the date from the timestamp
df['order_date'] = df['timestamp'].dt.date

#preview the new date column against the original timestamp column
df[['timestamp', 'order_date']].head(10)

#group the data by date while counting the number of orders
df_aggregated = pd.DataFrame(
df.groupby('order_date')['Order ID'].count()
).reset_index()

#preview the results
df_aggregated.head()

#plot the results by date and count of orders
df_aggregated.plot.line(x='order_date', y='Order ID')

#create a day of the week column, this is a numeric value
df_aggregated['day_of_week'] = pd.to_datetime(df_aggregated['order_date']).dt.weekday

#create a day name column
df_aggregated['day_name'] = pd.to_datetime(df_aggregated['order_date']).dt.day_name()

#group the data by the day of the week
df_weekdays = pd.DataFrame(df_aggregated.groupby(['day_of_week', 'day_name'])['Order ID'].mean()).reset_index()

#preview the results
df_weekdays.head(10)

import pandas as pd
import matplotlib.pyplot as plt

#create bar graph for orders by the day of the week
ax = df_weekdays.plot(x='day_name', y='Order ID', kind='bar')

#display the bar graph
plt.show()

#create data frame to see how many orders each customer has made
df_customer_frequency = pd.DataFrame(df.groupby('Customer ID')['Order ID'].count()).reset_index()

#preview the results
df_customer_frequency.head()

#specify how many bins there should be
bin_vals = list(range(0, 26))

#create histogram for customer order frequency
df_customer_frequency['Order ID'].hist(bins = bin_vals)

#identify how many customers in total there are
all_customers = len(df_customer_frequency)

#identify how many customers at more than one order
one_order_customers = len(
df_customer_frequency[df_customer_frequency['Order ID'] > 1]
)

#display total number of customers
print(all_customers)

#display the customers with more than 1 order
print(one_order_customers)

#calculate and display the percent of customers with more than one order
print(one_order_customers / all_customers)

#calculate the row number
df['row_num_join'] = df.sort_values(by=['timestamp'], ascending = False).groupby(['Customer ID']).cumcount() + 1

#preview the data where the row_num_join is more than 1
df.loc[df['row_num_join'] > 1, ['timestamp', 'Customer ID', 'row_num_join']].head(10)

#see customer values based on index
print(df['Customer ID'][5])
print(df['Customer ID'][11])

#create list of the two customer examples
customer_id_list = [df['Customer ID'][5], df['Customer ID'][11]]

#preview of results for only specific columns and the customers in our list
df.loc[df['Customer ID'].isin(customer_id_list), ['timestamp', 'Customer ID', 'row_num_join']].sort_values(by=['Customer ID' ,'timestamp'], ascending = True)

#select only the necessary columns for the join
df_for_join = df[['timestamp', 'Customer ID', 'row_num_join']]

#adjust the row number to be one less
df_for_join['row_num_join'] = df_for_join['row_num_join'] – 1

#join the data on customer ID and row number
df_joined = pd.merge(df, df_for_join, how = 'left', on = ['Customer ID','row_num_join'])

#print rows and columns for non-NaN values
print(df_joined[df_joined['timestamp_y'].notna()].shape)

#print the shape for all values
print(df_joined.shape)

#create time difference column
df_joined['time_difference'] = df_joined['timestamp_y'] - df_joined['timestamp_x']

#create day difference column
df_joined['day_difference'] = df_joined['time_difference'].dt.days

#preview only select columns to validate results
df_joined[['Customer ID', 'timestamp_x', 'timestamp_y', 'time_difference', 'day_difference']].head()

#specify the bin size of 7 days
bin_vals = list(range(-6, 57, 7))

#print the bin values to verify
print("bins:", bin_vals)

#print how many orders there were within a week
print("orders within 1 week: ", df_joined[df_joined['day_difference'] <= 7].shape[0])

#create histogram of the day differences
df_joined['day_difference'].hist(bins = bin_vals)
