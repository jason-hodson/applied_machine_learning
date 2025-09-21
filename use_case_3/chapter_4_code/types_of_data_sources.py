#read in first excel file
df_1 = pd.read_excel("Crime_Data_from_2020_to_Present_1.xlsx")

#read in second excel file
df_2 = pd.read_excel("Crime_Data_from_2020_to_Present_2.xlsx")

#stack the two datasets on top of each other
df = pd.concat([df_1, df_2], axis=0)

#save the dataset to .csv
df.to_csv('crime_data.csv', index=False)

#test the time to read in the .csv file
df_test = pd.read_csv("crime_data.csv")
