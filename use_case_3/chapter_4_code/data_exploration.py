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
