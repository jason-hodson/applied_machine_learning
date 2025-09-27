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



####Dummy Coding

#print unique values for each column
for col in df.columns:
    print(col, ": ", len(df[col].unique()))

#look at unique values in distance
df['Distance'].unique()

#define function to convert the distance from a string to numeric column
def convert_distance(col):
    if col == '<1km':
        return 0.5
    else:
        return col[:-2]

#apply the function created to convert distance column from string to numeric
df['DistanceNumeric'] = df['Distance'].apply(convert_distance)

#check the results
df.groupby(['Distance', 'DistanceNumeric'])['timestamp'].count()

#drop the original distance column
df = df.drop(columns=['Distance'])

#look at unique values in customer complaint tag column
df['Customer complaint tag'].unique()

#check unique values in cancellation/rejection reason
df['Cancellation / Rejection reason'].unique()

#check unique values in order ready marked column
df['Order Ready Marked'].unique()

#create list of columns to drop
cols_to_drop = ['Restaurant ID', 'City', 'Order ID', 'Delivery', 'Order Placed At','Items in order', 'Discount construct', 'Order Status']

#drop columns
df = df.drop(columns = cols_to_drop)

#create list of columns to dummy code
cols_to_dummy = ['Restaurant name', 'Subzone', 'Cancellation / Rejection reason', 'Order Ready Marked', 'Customer complaint tag']

#dummy code columns
df_dummied = pd.get_dummies(df, columns=cols_to_dummy)

