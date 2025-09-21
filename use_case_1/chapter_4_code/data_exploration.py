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
