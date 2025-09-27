####Replacing missing values

#drop the columns with many missing values
df = df.drop(columns=['Rating','Restaurant compensation (Cancellation)','Restaurant penalty (Rejection)'])

#preview the results of the number of KPT duration missing values
df[df['KPT duration (minutes)'].isna()].groupby('order_date')['Customer ID'].count().reset_index().sort_values('Customer ID', ascending = False).head(20)

#print the mean and median of KPT duration 
print(df['KPT duration (minutes)'].mean())
print(df['KPT duration (minutes)'].median())

#print the mean and median of rider wait time
print(df['Rider wait time (minutes)'].mean())
print(df['Rider wait time (minutes)'].median())

#calculate median of kpt duration and rider wait time
kpt_duration_median = df['KPT duration (minutes)'].median()
rider_wait_median = df['Rider wait time (minutes)'].median()

#replace missing values for kpt duration and rider wait time with the median
df['KPT duration (minutes)'] = df['KPT duration (minutes)'].fillna(kpt_duration_median)
df['Rider wait time (minutes)'] = df['Rider wait time (minutes)'].fillna(rider_wait_median)

#check for any remaining missing values in kpt duration or rider wait time
print(df['KPT duration (minutes)'].isna().any())
print(df['Rider wait time (minutes)'].isna().any())

#create list of missing values by column
for col in df.columns:
    print(col, ": ", df[col].isna().sum())

#drop the instructions and review columns
df = df.drop(columns=['Instructions','Review'])
