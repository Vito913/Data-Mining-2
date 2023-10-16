import os
import pandas as pd

# Define the path to your data folder
data_path = "data/op_spam_v1.4/negative_polarity"

# Initialize empty lists to store the data for train and test sets
train_data = []
test_data = []

# List of sub-subfolders (fold1 to fold5)
sub_subfolders = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']

# Loop through the sub-subfolders
for sub_subfolder in sub_subfolders:
    # Define the full path to the sub-subfolder for deceptive
    deceptive_subfolder_path = os.path.join(data_path, 'deceptive_from_MTurk', sub_subfolder)
    # Define the full path to the sub-subfolder for truthful
    truthful_subfolder_path = os.path.join(data_path, 'truthful_from_Web', sub_subfolder)
    
    # Loop through the files in the sub-subfolders
    for folder_path in [deceptive_subfolder_path, truthful_subfolder_path]:
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            
            # Read the contents of the file
            with open(file_path, 'r', encoding='latin1') as f:
                content = f.read()
                
                # Append to the appropriate list based on sub_subfolder (fold5 goes to test set)
                if sub_subfolder == 'fold5':
                    test_data.append({'Content': content, 'Subfolder': os.path.basename(folder_path)})
                else:
                    train_data.append({'Content': content, 'Subfolder': os.path.basename(folder_path)})

# Create DataFrames
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

# Display the DataFrames
print("Train Set:")
print(train_df.head(10),len(train_df))

print("\nTest Set:")
print(test_df.head(10),len(test_df))
