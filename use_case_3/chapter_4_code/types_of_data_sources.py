import pandas as pd
import os 

#execute for loop on the set of code
for file_name in os.listdir():

	#specifically only reference the .xlsx files in the directory
	if file_name.endswith(“.xlsx”):

		#read in the file as a generic object name
		df_temp = pd.read_excel(file_name)

		#stack each file on top of each other into what was a blank df
		df = pd.concat([df, df_temp], axis = 0)


#save the dataset to .csv
df.to_csv('crime_data.csv', index=False)

#test the time to read in the .csv file
df_test = pd.read_csv("crime_data.csv")
