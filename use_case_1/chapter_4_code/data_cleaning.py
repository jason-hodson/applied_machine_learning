####Dummy Coding

#load in necessary libraries
import pandas as pd
import os
import datetime

#create blank dataframe
df = pd.DataFrame()

#create loop to go through all files in the current directory
for file_name in os.listdir():

    #set up if statement to look at only .csv files
    if file_name.endswith(".csv"):
	 
	  #read in the csv file into a temp df
        df_temp = pd.read_csv(file_name, encoding='unicode_escape')

	  #stack the data frames on top of each other
        df = pd.concat([df, df_temp], axis = 0)

#replace the hyphens with backslashes for the date
df['new_invoicedate'] = df['InvoiceDate'].str.replace('-','/')

#convert the date column to a pandas date time
df['new_invoicedate'] = pd.to_datetime(df['new_invoicedate'],format = '%d/%m/%Y %H:%M')

#create a date for from the timestamp
df['invoicedate'] = df['new_invoicedate'].dt.date

#keep only the data identified as legitimate transaction records
df_cleaned = df[(df['Quantity'] > 0) & (df['Price'] > 0) & (df['Customer ID'] > 0)]

#preview data
df_cleaned.head()

#print unique valeus in description column
print("Unique values in description column: ", df_cleaned['Description'].nunique())

#print unique values in customer ID column
print("Unique values in customer ID column: ", df_cleaned['Customer ID'].nunique())

#print unique values in country column
print("Unique values in country column: ", df_cleaned['Country'].nunique())

#dummy code the country column
df_country = pd.get_dummies(df_cleaned[['Country']], dtype=float)

#print the number of columns and rows
print(df_country.shape)

#preview the data
df_country.head()

#dummy code the description column
df_description = pd.get_dummies(df_cleaned[['Description']], dtype=float)

#aggregate the data by description, counting the number of records
df_summarized = pd.DataFrame(df_cleaned.groupby('Description')['Invoice'].count()).reset_index()

#sort the values by the descriptions that have the most invoices
df_summarized = df_summarized.sort_values(by='Invoice', ascending=False)

#preview the data
df_summarized.head()

#use head and tail to select specific range to preview
df_summarized.head(100).tail(10)

#use head and tail to select specific range to preview
df_summarized.head(50).tail(10)

#select the top 50 description values
df_description_top_50 = df_summarized.head(50)

#Rename colummn to description grouped
df_description_top_50['Description Grouped'] = df_description_top_50['Description']

#merge onto the original dataset
df_cleaned = df_cleaned.merge(df_description_top_50[['Description', 'Description Grouped']], left_on = 'Description', right_on = 'Description')

#preview data
df_cleaned.head()

#dummy code the grouped description column
df_description = pd.get_dummies(df_cleaned[['Description Grouped']], dtype=float)

#print the number of rows and columns
print(df_description.shape)

#preview the data
df_description.head()

#count the number of instances each customer appears in the data
df_summarized = pd.DataFrame(df_cleaned.groupby('Customer ID')['Invoice'].count()).reset_index()

#sort from most to least
df_summarized = df_summarized.sort_values(by='Invoice', ascending=False)

#use head and tail to preview specific range of the data
df_summarized.head(100).tail(10)

#select top 100 values
df_customer_top_100 = df_summarized.head(100)

#rename the column
df_customer_top_100['Customers Grouped'] = df_customer_top_100['Customer ID']

#merge onto original dataset
df_cleaned = df_cleaned.merge(df_customer_top_100[['Customer ID', 'Customers Grouped']], left_on = 'Customer ID', right_on = 'Customer ID')

#dummy code the customer data
df_customers = pd.get_dummies(df_cleaned[['Customers Grouped']], dtype=float, dummy_na = True)

#print the number of rows and columns
print(df_customers.shape)

#preview the data
df_customers.head()

#updated code for dummy coding the customer column
df_customers = pd.get_dummies(df_cleaned[['Customers Grouped']].astype(str), dtype=float, dummy_na = True)

#print the number of rows and columns
print(df_customers.shape)

#preview the data
df_customers.head()

#select the top 50 values
df_description_top_50 = df_summarized.head(50)

#rename column
df_description_top_50['Description Grouped'] = df_description_top_50['Description']

#updated merge using the how parameter
df_cleaned = df_cleaned.merge(df_description_top_50[['Description', 'Description Grouped']], how = ‘left’,left_on = 'Description', right_on = 'Description')

#selecting top 100 values
df_customer_top_100 = df_summarized.head(100)

#renaming columns
df_customer_top_100['Customers Grouped'] = df_customer_top_100['Customer ID']

#updated merge using how parameter
df_cleaned = df_cleaned.merge(df_customer_top_100[['Customer ID', 'Customers Grouped']], how = ’left’left_on = 'Customer ID', right_on = 'Customer ID')

#dummy code the description column
df_description = pd.get_dummies(df_cleaned[['Description Grouped']], dtype=float)

#print number of rows and columns
print(df_description.shape)

#preview the data
df_description.head()

#dummy code the customer data
df_customers = pd.get_dummies(df_cleaned[['Customers Grouped']].astype(str), dtype=float, dummy_na = True)

#print number of rows and columns
print(df_customers.shape)

#preview the data
df_customers.head()

