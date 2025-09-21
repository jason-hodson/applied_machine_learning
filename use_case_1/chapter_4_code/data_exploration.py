####Data Types####


#see the data types of the data frame
df.dtypes

#create new column called new_invoicedate with properly formatted date
df[‘new_invoicedate’] = pd.to_datetime(df[‘InvoiceDate’], format = ‘%d/%m/%Y %H:%M’)

#replace ‘-‘ with ‘/’ in the column
df[‘new_invoicedate’] = df[‘InvoiceDate’].str.replace(‘-‘, ’/’)

#create the new column with the properly formatted date
df[‘new_invoicedate’] = pd.to_datetime(
  df[‘new_invoicedate’], 
  format = ‘%d/%m/%Y %H:%M’)

#preview the output
df.head()

#import the datetime library, a library used for date and timestamp columns
import datetime

#extract just the date from the timestamp
df[‘invoicedate’] = df[‘new_invoicedate’].dt.date

#preview the output
df.head()


####Data Visualization####

#group/summarize the data by date, while summing price
#resetting the index ensures the column names of the output can be referenced 
df_summarized = pd.DataFrame(df.groupby(‘invoicedate’)[‘Price’].sum()).reset_index()

#preview the data
df_summarized.head()

#import matplotlib the most common plotting library in Python
import matplotlib.pyplot as plt

#plot a line graph based on the date and price
df.plot.line(x=’invoicedate’, y=’Price’)

#check the data where price is negative
df[df[‘Price’] < 0

#apply filters based on the information from Chris and your exploration
df_cleaned = df[
  (df[‘Quantity’] > 0) & 
  (df[‘Price’] > 0) & 
  (df[‘Customer ID’] > 0)
]

#group the dataframe again by date and summing price
df_summarized = pd.DataFrame(df_cleaned.groupby(‘invoicedate’)[‘Price’].sum())

#reset the index to ensure the columns of the output can be referenced
df_summarized = df_summarized.reset_index()

#plot the data by date and price
df_summarized.plot.line(x=’invoicedate’, y=’Price’)

