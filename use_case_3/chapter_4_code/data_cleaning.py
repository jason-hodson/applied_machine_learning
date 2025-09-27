####Replacing missing values

#loop through each column of the data frame
for col in df.columns:
    #print out the total number of na/blank values in the dat frame
    print(col, ": ", df[col].isna().sum())


####Dummy coding

#count number of unique values in the column
print(df['AREA NAME'].nunique())

#create new data frame that counts the number of crimes for each area name
#sorts the values from largest to smallest
df_review = df.groupby('AREA NAME')['DR_NO'].count().reset_index().sort_values('DR_NO', ascending = False)

#displays the entirety of the data frame
df_review

#count number of unique values in the column
print(df['Crm Cd Desc'].nunique())

#create new data frame that counts the number of crimes for each crime code
#sorts the values from largest to smallest
df_review = df.groupby('Crm Cd Desc')['DR_NO'].count().reset_index().sort_values('DR_NO', ascending = False)

#displays the first 20 rows of the data frame
df_review.head(20)

#create list of the top 15 unique values based on count of the Crm Cd Desc
top_15_crime_codes = list(df_review.head(15)['Crm Cd Desc'].unique())

#create new column that either shows the top 15 crime code or other
df['crime_description'] = np.where(df['Crm Cd Desc'].isin(top_15_crime_codes), df['Crm Cd Desc'], 'Other')

#count number of unique values in the column
print(df['Vict Sex'].nunique())

#create new data frame that counts the number of crimes for each value
#sorts the values from largest to smallest
df_review = df.groupby('Vict Sex')['DR_NO'].count().reset_index().sort_values('DR_NO', ascending = False)

#displays all rows of the data frame
df_review

#create list of male and female values
gender_list_keep = ['M', 'F']

#create new column that groups the values that aren’t M or F into an Other value
df['victim_gender'] = np.where(df['Vict Sex'].isin(gender_list_keep), df['Vict Sex'], 'Other')

#count number of unique values in the column
print(df['Vict Descent'].nunique())

#create new data frame that counts the number of crimes for each value
#sorts the values from largest to smallest
df_review = df.groupby('Vict Descent')['DR_NO'].count().reset_index().sort_values('DR_NO', ascending = False)

#displays all rows of the data frame
df_review

#list of values to keep
race_list_keep = ['H', 'W', 'B', 'X', 'O']

#create new race column that is consolidated
df['victim_race'] = np.where(df['Vict Descent'].isin(race_list_keep), df['Vict Descent'], 'Other')

#count number of unique values in the column
print(df['Premis Desc'].nunique())

#create new data frame that counts the number of crimes for each value
#sorts the values from largest to smallest
df_review = df.groupby('Premis Desc')['DR_NO'].count().reset_index().sort_values('DR_NO', ascending = False)

#displays top 20 values of the data frame
df_review.head(20)

#Create list of values based on the data frame we already created
top_7_crime_locations = list(df_review.head(15)['Premis Desc'].unique())

#Create the new column to consolidate the values
df['crime_premises'] = np.where(df['Premis Desc'].isin(top_7_crime_locations), df['Premis Desc'], 'Other')

#count number of unique values in the column
print(df['Weapon Desc'].nunique())

#create new data frame that counts the number of crimes for each value
#sorts the values from largest to smallest
df_review = df.groupby('Weapon Desc')['DR_NO'].count().reset_index().sort_values('DR_NO', ascending = False)

#displays top 20 values of the data frame
df_review.head(20)

#create list of the top 4 weapons by volume
top_4_weapons = list(df_review.head(4)['Weapon Desc'].unique())

#create new consolidated column that groups weapons values
df['crime_weapon'] = np.where(df['Weapon Desc'].isin(top_4_weapons), df['Weapon Desc'], 'Other')


#list of columns to dummy code
cols_to_dummy = ['AREA NAME', 'crime_description', 'victim_gender', 'victim_race', 'crime_premises', 'crime_weapon']

#execute dummy coding
df_dummied = pd.get_dummies(df, columns = cols_to_dummy)

#drop columns no longer needed
df_dummied = df_dummied.drop(
columns = ['AREA', 'Rpt Dist No', 'Part 1-2', 'Crm Cd', 'Crm Cd Desc', 'Mocodes','Vict Sex', 'Vict Descent','Premis Cd', 
           'Premis Desc', 'Weapon Used Cd', 'Weapon Desc', 'Status','Status Desc', 'Crm Cd 1', 'Crm Cd 2', 'Crm Cd 3', 
           'Crm Cd 4','LOCATION', 'Cross Street', 'LAT', 'LON'])

#save data to csv file
df_dummied.to_csv('crime_data_prepped.csv', index=False)
