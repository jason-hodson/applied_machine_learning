####Replacing missing values

#loop through each column of the data frame
for col in df.columns:
    #print out the total number of na/blank values in the dat frame
    print(col, ": ", df[col].isna().sum())
