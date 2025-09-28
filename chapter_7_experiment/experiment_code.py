import pandas as pd

#read in data
df = pd.read_csv("Experiment Data.csv")

#preview data
df.head()

#treatment group
df_natural_treatment_model = df[(df['Tariff Exposure']==1) & (df['Model Adjusting Offering']==1)]

#control group
df_natural_control_model = df[(df['Tariff Exposure']==0) & (df['Model Adjusting Offering']==1)]

#print the rows and columns of the data
print(df_natural_treatment_model.shape)
print(df_natural_control_model.shape)


#treatement group
df_natural_treatment_nomodel = df[(df['Tariff Exposure']==1) & (df['Model Adjusting Offering']==0)]

#control group
df_natural_control_nomodel = df[(df['Tariff Exposure']==0) & (df['Model Adjusting Offering']==0)]

#see number of rows and columns
print(df_natural_treatment_nomodel.shape)
print(df_natural_control_nomodel.shape)


#calculate the mean for each dataset
print(df_natural_treatment_model['Purchases in Next Month'].mean())
print(df_natural_control_model['Purchases in Next Month'].mean())
print(df_natural_treatment_nomodel['Purchases in Next Month'].mean())
print(df_natural_control_nomodel['Purchases in Next Month'].mean())

#import stats library
from scipy import stats

#run t test on data
t_statistic, p_value = stats.ttest_ind(df_natural_treatment_model['Purchases in Next Month'], df_natural_control_model['Purchases in Next Month'])

#print the p-value
print("P-value:", p_value)

#run t test on data
t_statistic, p_value = stats.ttest_ind(df_natural_treatment_nomodel['Purchases in Next Month'],df_natural_control_nomodel['Purchases in Next Month'])

#print the p-value
print("P-value:", p_value)


#filter to only the treatment group
df_treatment_full = df[df['Model Adjusting Offering'] == 1] 

#select only specific columns
df_treatment = df_treatment_full[[
'Previous Month Purchases', 
'Company Size', 
'Tariff Exposure'
]]

#filter to only the control group
df_control_full = df[df['Model Adjusting Offering'] == 0] 

#select only specific columns
df_control = df_control_full[[
'Previous Month Purchases', 
'Company Size', 
'Tariff Exposure'
]]


#create function to normalize the data
def normalize_dataframe(df):
  """Normalizes each column of a Pandas DataFrame to the range [0, 1].

  Args:
    df: The input Pandas DataFrame.

  Returns:
    A new DataFrame with normalized columns.
  """
  return df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))


#apply normalization function to each treatment and control
df_treatment = normalize_dataframe(df_treatment)
df_control = normalize_dataframe(df_control)


#import pairwise_distances function
from sklearn.metrics.pairwise import pairwise_distances

#create blank dataframe to add matches to
all_matches = pd.DataFrame()

#loop through treatment list to find matches
for treatment_row in list(range(0, len(df_treatment))):
    treatment = df_treatment.iloc[treatment_row].to_numpy().reshape(1, -1)
    treatment_company = df_treatment_full.iloc[treatment_row, 0]
    distances_list = list()
    for control_row in list(range(0, len(df_control))):
        control = df_control.iloc[control_row].to_numpy().reshape(1,-1)
        distance = pairwise_distances(treatment, control, metric = 'euclidean')[0][0]
        distances_list.append(distance)
    data = {
'Treatment Company': treatment_company * len(distances_list), 
'Control Company': df_control_full['Company'], 
'Distance': distances_list
}
    temp_df = pd.DataFrame(data)
    all_matches = pd.concat([all_matches, temp_df], ignore_index = True)


#load random library
import random

#create list of the treament companies
treatment_companies = all_matches['Treatment Company'].unique().tolist()

#randomize the order of the treatment companies
random.shuffle(treatment_companies)


#test with subset of the companies
for treatment_company in treatment_companies[0:2]:
    temp_df = all_matches_for_filtering[all_matches_for_filtering['Treatment Company']==treatment_company]
    min_distance_row = temp_df[temp_df['Distance'] == temp_df['Distance'].min()]
    print(min_distance_row)
    print(min_distance_row['Control Company'].iloc[0])
    print(min_distance_row['Distance'].iloc[0])

#rename data frame
all_matches_for_filtering = all_matches

#run test with subset of companies
for treatment_company in treatment_companies[0:2]:
    temp_df = all_matches_for_filtering[all_matches_for_filtering['Treatment Company']==treatment_company]
    min_distance_row = temp_df[temp_df['Distance'] == temp_df['Distance'].min()]
    print(min_distance_row)
    print(min_distance_row['Control Company'].iloc[0])
    print(min_distance_row['Distance'].iloc[0])
all_matches_for_filtering = all_matches_for_filtering[all_matches_for_filtering['Control Company'] != min_distance_row['Control Company'].iloc[0]]



#create new df
all_matches_for_filtering = all_matches

#create blank lists
treatment_company_vals = []
control_company_vals = []
distance_vals = []

#test on subset of companies
for treatment_company in treatment_companies[0:2]:
  temp_df = all_matches_for_filtering[all_matches_for_filtering['Treatment Company']==treatment_company]
  min_distance_row = temp_df[temp_df['Distance'] == temp_df['Distance'].min()]
  print(min_distance_row)
  print(min_distance_row['Control Company'].iloc[0])
  print(min_distance_row['Distance'].iloc[0])

	treatment_company_vals.append(treatment_company)
  control_company_vals.append(min_distance_row['Control Company'].iloc[0])
  distance_vals.append(min_distance_row['Distance'].iloc[0])
  	
  all_matches_for_filtering = all_matches_for_filtering[all_matches_for_filtering['Control Company'] != min_distance_row['Control Company'].iloc[0]]

#add lists into dictionary
data = {'Treatment Company': treatment_company_vals,'Control Company': control_company_vals, 'Distance': distance_vals}

#create a data frame from the dictionary
final_matched_df = pd.DataFrame(data)



#create function to match treatment and control together
def create_one_to_one_match(df):
    import random

    treatment_companies = df['Treatment Company'].unique().tolist()
    random.shuffle(treatment_companies)
    
    all_matches_for_filtering = df
    treatment_company_vals = []
    control_company_vals = []
    distance_vals = []
    
    for treatment_company in treatment_companies:
        temp_df = all_matches_for_filtering[all_matches_for_filtering['Treatment Company']==treatment_company]
        min_distance_row = temp_df[temp_df['Distance'] == temp_df['Distance'].min()]
        #print(min_distance_row)
        #print(min_distance_row['Control Company'].iloc[0])
        #print(min_distance_row['Distance'].iloc[0])
    
        treatment_company_vals.append(treatment_company)
        control_company_vals.append(min_distance_row['Control Company'].iloc[0])
        distance_vals.append(min_distance_row['Distance'].iloc[0])
        
        all_matches_for_filtering = all_matches_for_filtering[all_matches_for_filtering['Control Company'] != min_distance_row['Control Company'].iloc[0]]
    
    data = {'Treatment Company': treatment_company_vals,'Control Company': control_company_vals, 'Distance': distance_vals}
    
    final_matched_df = pd.DataFrame(data)

    return final_matched_df



#create list of values from 1 to 10
iterations = list(range(1, 11, 1))

#create large number
winning_distance = 90000*90000

#create loop to run the matching function 10 times to find best matching result
for iter in iterations:
    match_iteration = create_one_to_one_match(all_matches)
    match_distance = sum(match_iteration['Distance'])
    if match_distance < winning_distance:
        winning_df = match_iteration
        print(match_distance, winning_distance)
        print("New best match score")
        winning_distance = match_distance
    else:
        print("Best match score remains")



#filter to matched control data
df_control_matched = df_control_full[df_control_full['Company'].isin(winning_df['Control Company'].unique())]

#print mean of the treatment and control groups
print(df_control_matched['Purchases in Next Month'].mean())
print(df_treatment_full['Purchases in Next Month'].mean())


#load stats library
from scipy import stats

#run t test on data
t_statistic, p_value = stats.ttest_ind(df_control_matched['Purchases in Next Month'], df_treatment_full['Purchases in Next Month'])

#print the p-value
print("P-value:", p_value)
