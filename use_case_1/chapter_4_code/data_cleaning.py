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


####Dimensionality Reduction

#create list of values for components to try
versions_of_n = [5, 10, 15, 20, 25]

#loop to try each iteration of components
for n in versions_of_n:
    #create pca object
    pca = PCA(n_components=n)

    #fit pca to df_country
    pca.fit(df_country)

    #print explained ratios of the components
    print("Explained variance ratio:", pca.explained_variance_ratio_)


#pick the 11 components
pca_country = PCA(n_components=11)

#fit the pca model to df_country
pca_country.fit(df_country)

#print the explained variance ratio
print("Explained variance ratio:", sum(pca_country.explained_variance_ratio_))

#using 50 components for the description field
pca_description = PCA(n_components=50)

#fit the pca model to df_description
pca_description.fit(df_description)

#print the explained variance ratio
print("Explained variance ratio:", pca_description.explained_variance_ratio_)

#use 50 for the components on customer
pca_customer = PCA(n_components=50)

#fit the pca model to df_customers
pca_customer.fit(df_customers)

#print the explained variance ratio
print("Explained variance ratio:", pca_customer.explained_variance_ratio_)

#pick number of components
versions_of_n = [5, 10, 15]

#for loop to try various iterations of components
for n in versions_of_n:

    #create inputs for LDA
    x = df_country

    #create target variable for LDA
    y = df_cleaned['Price'].astype(int)
    
    #split into training and test data
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    
    #create lda model with number of components from the loop
    lda = LinearDiscriminantAnalysis(n_components=n)

    #fit the LDA model
    lda.fit(X_train, y_train)
    
    #predict on the test dataset with the lda model
    y_pred = lda.predict(X_test)
    
    #create the accuracy score
    accuracy = accuracy_score(y_test, y_pred)

    #print the accuracy score
    print(accuracy)




