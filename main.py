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
                    test_data.append({'Content': content})
                else:
                    train_data.append({'Content': content})

# Create DataFrames
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

# Display the DataFrames
print("Train Set:")
print(train_df.head(10),len(train_df))

print("\nTest Set:")
print(test_df.head(10),len(test_df))

# Path to the data folder
data_folder = 'data/op_spam_v1.4'

# Specify the first subfolder (neg)
neg_subdirectory_path = os.path.join(data_folder, 'negative_polarity')

# Initialize empty lists to store file paths and labels
fold_1_to_4_files = []
fold_5_files = []
labels_1_to_4 = []
labels_5 = []

# Loop through the 2 subdirectories inside 'neg'
for subdir in os.listdir(neg_subdirectory_path):
    subdir_path = os.path.join(neg_subdirectory_path, subdir)
    for fold in os.listdir(subdir_path):
        fold_path = os.path.join(subdir_path, fold)
        # Check if the item in the directory is a directory and starts with 'fold'
        if os.path.isdir(fold_path) and fold.startswith('fold'):
            for root, dirs, files in os.walk(fold_path):
                for file in files:
                    label = 0 if file[0].lower() == 't' else 1
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        file_content = file.read()
                        if fold != 'fold5':
                            fold_1_to_4_files.append(file_content)
                            labels_1_to_4.append(label)
                        else:
                            fold_5_files.append(file_content)
                            labels_5.append(label)

# Create dataframes from the gathered file paths and labels
df_train = pd.DataFrame({'Review Text': fold_1_to_4_files, 'Label': labels_1_to_4})
df_test = pd.DataFrame({'Review Text': fold_5_files, 'Label': labels_5})

print(labels_1_to_4)
print(labels_5)

# Print first few rows of the dataframes
print('Dataframe for fold1 to fold4:')
print(df_train)

print('Dataframe for fold5:')
print(df_test)

