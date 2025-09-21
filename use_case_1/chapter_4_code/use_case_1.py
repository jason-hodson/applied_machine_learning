#load pandas library
import pandas as pd

#load os library, used specifically for navigating file paths
import os

#read in the csv file
df_1 = pd.read_csv(“Retail Sales File 1.csv”, encoding=’unicode_escape’)
