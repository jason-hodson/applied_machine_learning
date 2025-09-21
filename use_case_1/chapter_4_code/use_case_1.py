#load pandas library
import pandas as pd

#load os library, used specifically for navigating file paths
import os

#read in the csv file
df_1 = pd.read_csv(“Retail Sales File 1.csv”, encoding=’unicode_escape’)

#create blank data frame
df = pd.DataFrame()

#execute for loop on the set of code
for file_name in os.listdir():

	#specifically only reference the .csv files in the directory
	if file_name.endswith(“.csv”):

		#read in the file as a generic object name
		df_temp = pd.read_csv(file_name, encoding = ‘unicode_escape’)

		#stack each file on top of each other into what was a blank df
		df = pd.concat([df, df_temp], axis = 0)
