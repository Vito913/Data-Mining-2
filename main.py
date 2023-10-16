'''
import os

# Define the path to your data folder
data_path = "C:/Users/PC/OneDrive/Desktop/Data-Mining-2/data/op_spam_v1.4"
# Define the subfolder you want to access
subfolder = 'negative_polarity'

# Define the full path to the subfolder
subfolder_path = os.path.join(data_path, subfolder)

# Check if the subfolder exists
if os.path.exists(subfolder_path):
    print("Hello")
else:
    print(f"The subfolder '{subfolder}' does not exist.")
import os

# Define the path to your data folder
data_path = "C:/Users/PC/OneDrive/Desktop/Data-Mining-2/data/op_spam_v1.4"

# Define the subfolder you want to access
subfolder = 'negative_polarity'

# Define the full path to the subfolder
subfolder_path = os.path.join(data_path, subfolder)

# Check if the subfolder exists
if os.path.exists(subfolder_path):
    print("Hello")
else:
    print(f"The subfolder '{subfolder}' does not exist.")
'''

import os
import pandas as pd

# Define the path to your data folder
data_path = "C:/Users/PC/OneDrive/Desktop/Data-Mining-2/data/op_spam_v1.4"

# Define the subfolders you want to access
subfolders = ['negative_polarity', 'positive_polarity']

# Initialize an empty list to store the data
data = []

# List of sub-subfolders (fold1 to fold4)
sub_subfolders = ['fold1', 'fold2', 'fold3', 'fold4']

# Loop through the subfolders
for subfolder in subfolders:
    # Define the full path to the subfolder
    subfolder_path = os.path.join(data_path, subfolder)
    
    # Loop through the sub-subfolders
    for sub_subfolder in sub_subfolders:
        # Define the full path to the sub-subfolder
        sub_subfolder_path = os.path.join(subfolder_path, 'deceptive_from_MTurk', sub_subfolder)
        
        # Loop through the files in the sub-subfolder
        for file in os.listdir(sub_subfolder_path):
            file_path = os.path.join(sub_subfolder_path, file)
            
            # Read the contents of the file and append to the data list
            with open(file_path, 'r', encoding='latin1') as f:
                content = f.read()
                data.append({'Subfolder': subfolder, 'Content': content})

# Create a DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
print(len(df))
