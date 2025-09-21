####Data Types####


#convert date rptd column into the pandas datetime format
df['Date Rptd'] = pd.to_datetime(df['Date Rptd'])

#backfill the time column to ensure it has at least enough values
df['TIME OCC'] = df['TIME OCC'].astype(str).str.zfill(4)

#extract the necessary components for a timestamp
df['TIME OCC'] = df['TIME OCC'].str[:2] + ':' + df['TIME OCC'].str[2:]

#concatenate the date and time together to creat a timestamp
df['TIMESTAMP'] = pd.to_datetime(df['DATE OCC'] + ' ' + df['TIME OCC'])

#check only the date and time columns alongside the new timestamp column
df[['DATE OCC', 'TIME OCC', 'TIMESTAMP']]

#drop the original date and time columns
df = df.drop(columns=['DATE OCC','TIME OCC'])


####Data Visualization####
#create the date column
df['DATE'] = df['TIMESTAMP'].dt.date

#create new data frame representing number of crimes on each day
df_by_date = df.groupby('DATE')['DR_NO'].count().reset_index()

#plot the crime by day data frame
df_by_date.plot(kind='line', x = 'DATE', y = 'DR_NO')
plt.show()

#create new column to identify if date is before or after the start of 2024
df['before_after_2024'] = np.where(df['TIMESTAMP'] >= pd.to_datetime('2024-01-01'), "2024 or After", "Before 2024")

#group data by area and the new 2024 date indicator column
df_area_2024_split = df.groupby(['AREA NAME', 'before_after_2024'])['DR_NO'].count().reset_index()

#use the pivot function to make the before or after 2024 data as two columns side by side
df_area_2024_split = df_area_2024_split.pivot(index='AREA NAME', columns='before_after_2024', values='DR_NO').reset_index()

#create percent change column to use as primary point of comparison
df_area_2024_split['percent_change'] = round((df_area_2024_split['2024 or After'] - df_area_2024_split['Before 2024']) /df_area_2024_split['Before 2024'], 2)

#sort the values in descending order
df_area_2024_split = df_area_2024_split.sort_values('percent_change', ascending = False)

#set width of graph larger to account for number of areas
plt.figure(figsize=(20,8))

#create base bar graph
bar_plt = plt.bar(df_area_2024_split['AREA NAME'], df_area_2024_split['percent_change'])

#specify the limits of the y-axis since we know it’s based on a percentage
plt.ylim(-1,0)

#show the data label on the bar itself
plt.bar_label(bar_plt)

#angle the X-axis label and align to make it easier to read
plt.xticks(rotation=45, ha='right')

#display the bar plot
plt.show()

#create list of columns to visualize
col_list = ['AREA NAME', 'Crm Cd Desc', 'Vict Age', 'Vict Sex', 'Vict Descent', 'Premis Desc', 'Weapon Desc']

#for loop through the list of columns
for col in col_list:
    #group the data by the column, using the report number for counting
    group_df = df.groupby(col)['DR_NO'].count().reset_index()
    
    #create the bar graph
    plt.figure(figsize=(10,6))
    plt.bar(group_df[col], group_df['DR_NO'])
    plt.title(col)
    plt.xticks(rotation=45, ha='right')
    plt.show()

####Descriptive Statistics####


#print the median of victim age
print(df['Vict Age'].median())

#print the mean of the victim age
print(df['Vict Age'].mean())

#print the median victim age for ages greater than 0
print(df[df['Vict Age'] > 0]['Vict Age'].median())

#print the mean victim age for ages greater than 0
print(df[df['Vict Age'] > 0]['Vict Age'].mean())


