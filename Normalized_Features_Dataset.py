from Segmentation_FeaturesExtract import CSV
import numpy as np
import pandas as pd
import os
import json

#The file name of the directory 
file_path = 'heartSound_data.csv'

#Read the csv file
dataset_df = pd.read_csv(file_path)

#Initialize the CSV class
csv_document = CSV()
#Create a csv file call 
column_names, file_name = csv_document.create_csv_file(file='normalized_data.csv', createFile=True)

#Finding min and max
row_length = len(dataset_df)
column_length = len(column_names)-1

#Initialize the dictionary
data = {}
data['normalize_range'] = []

#Loop through the column 
for i in range(0, column_length):
    #Extract the maximum and minimum value of the column
    minimum_value = dataset_df[column_names[i]].min()
    maximum_value = dataset_df[column_names[i]].max()

    #write the minimum value and maximum value to dictionary
    data['normalize_range'].append({
        'minimum': minimum_value,
        'maximum': maximum_value
    })
    
    for j in range(0, row_length):
        #extract the row value in the column 
        value = dataset_df[column_names[i]][j]
        #Normalize the value
        normalize_value = (value - minimum_value)/(maximum_value - minimum_value)
        #Replace the value with normalized value
        dataset_df.loc[j, column_names[i]] = normalize_value
#write the normalized dataset to csv file 
dataset_df.to_csv(file_name, index=False)

#Write the data to json file
with open('ranges.json', 'w') as outfile:
    json.dump(data, outfile)
