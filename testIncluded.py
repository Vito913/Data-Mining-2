import os
import pandas as pd
import string
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from itertools import chain
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


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

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Checks if we are dropping a percentage of the terms
drop = True
# The amount of terms to drop
drop_percent = 0.0015 
# Checks if we are using part of speech tagging
part_of_speech = True
# Checks if we are using stemming
stemming = True


# Path to the data folder
data_folder = 'data/op_spam_v1.4'

# Specify the first subfolder (neg)
neg_subdirectory_path = os.path.join(data_folder, 'negative_polarity')

# Initialize empty lists to store file paths and labels
fold_1_to_4_files = []
fold_5_files = []
labels_1_to_4 = []
labels_5 = []
filenamesTrain = []
filenamesTest = []

# Loop through the 2 subdirectories inside 'neg'
for subdir in os.listdir(neg_subdirectory_path):
    subdir_path = os.path.join(neg_subdirectory_path, subdir)
    for fold in os.listdir(subdir_path):
        fold_path = os.path.join(subdir_path, fold)
        # Check if the item in the directory is a directory and starts with 'fold'
        if os.path.isdir(fold_path) and fold.startswith('fold'):
            for root, dirs, files in os.walk(fold_path):
                for file in files:
                    file_name = file
                    label = 0 if file[0].lower() == 't' else 1
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        file_content = file.read()
                        if fold != 'fold5':
                            fold_1_to_4_files.append(file_content)
                            labels_1_to_4.append(label)
                            filenamesTrain.append(file_name)
                        else:
                            fold_5_files.append(file_content)
                            labels_5.append(label)
                            filenamesTest.append(file_name)

# Create dataframes from the gathered file paths and labels
df_train = pd.DataFrame({'Review Text': fold_1_to_4_files, 'Label': labels_1_to_4})
df_test = pd.DataFrame({'Review Text': fold_5_files, 'Label': labels_5})
#print(filenamesTrain)
def remove_unnecessary(row):
    # Remove numbers
    row = ''.join([i for i in row if not i.isdigit()])
    
    # Remove punctuations
    row = row.translate(str.maketrans('', '', string.punctuation))
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    row = ' '.join([word for word in row.split() if word.lower() not in stop_words])
    
    # lowercase
    row = row.lower()
    
    # Remove unnneccesary whitespaces
    row = row.strip()
    
    return row

## implement stemming

def stemming(row):
    ps = PorterStemmer()
    words = word_tokenize(row)
    new_row = []
    for w in words:
        new_row.append(ps.stem(w))
    return ' '.join(new_row)


# Part of speech tagging

def pos_tagging(row):
    words = word_tokenize(row)
    return nltk.pos_tag(words)



# Print first few rows of the dataframes
#print('Dataframe for fold1 to fold4:')
#print(df_train)

#print('Dataframe for fold5:')
#print(df_test)

# Apply the remove_unnecessary function to each row in the df_train and df_test dataframes
df_train['Review Text'] = df_train['Review Text'].apply(remove_unnecessary).apply(stemming).apply(pos_tagging)
df_test['Review Text'] = df_test['Review Text'].apply(remove_unnecessary).apply(stemming).apply(pos_tagging)

# print(labels_1_to_4)
# print(labels_5)

# Print first few rows of the dataframes
#print('Dataframe for fold1 to fold4:')
#print(df_train)

#print('Dataframe for fold5:')
#print(df_test)


##remove sparse terms already done or not??

##create document-term matrix for train data -> rows = files and columns = terms

# Extract words without POS from the 'Review Text' column
words_only_reviews_train = [[' '.join(word for word, pos in review)] for review in df_train['Review Text']]
words_only_reviews_test = [[' '.join(word for word, pos in review)] for review in df_test['Review Text']]
# Flatten the list of words into a single list
all_words_train = list(chain.from_iterable(words_only_reviews_train))
all_words_test = list(chain.from_iterable(words_only_reviews_test))
# Convert the list of words into space-separated strings for each review
doc_strings_train = [' '.join(review) for review in words_only_reviews_train]
doc_strings_test = [' '.join(review) for review in words_only_reviews_test]
# Initialize the CountVectorizer
vectorizerTrain = CountVectorizer()
vectorizerTest = CountVectorizer()
# Transform the documents into a document-term co-occurrence matrix
doc_term_matrix_train = vectorizerTrain.fit_transform(doc_strings_train)
doc_term_matrix_test = vectorizerTest.fit_transform(doc_strings_test)
# Convert the document-term matrix to an array
doc_term_matrix_array_train = doc_term_matrix_train.toarray()
doc_term_matrix_array_test = doc_term_matrix_test.toarray()
if drop:
    doc_term_matrix_array_train = doc_term_matrix_array_train[:, (doc_term_matrix_array_train.sum(axis=0) >= drop_percent * doc_term_matrix_array_train.shape[0])]
    #also for test?? or should we just drop same words dropped in test?????????????

# Get the feature names (words) corresponding to the columns of the matrix
feature_names_train = vectorizerTrain.get_feature_names_out()
feature_names_test = vectorizerTest.get_feature_names_out()
## Check which words happen less than drop_percent times and remove them from the matrix
docTermMatrixTrain = pd.DataFrame(doc_term_matrix_array_train, columns=feature_names_train, index=filenamesTrain)
print(docTermMatrixTrain)

docTermMatrixTest = pd.DataFrame(doc_term_matrix_array_test, columns=feature_names_test, index=filenamesTest)
print(docTermMatrixTest)


param_grid = {
    'n_estimators': [200, 400, 500, 600, 700, 800, 1000],  # List of different numbers of trees
    'max_features': ['sqrt', 'log2', None]  # Different options for max_features
}
cv = KFold(n_splits=10, shuffle=True, random_state=42) 
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, n_jobs=-1)
history = grid_search.fit(doc_term_matrix_array_train, df_train['Label'])

best_n_estimators = grid_search.best_params_['n_estimators']
best_max_features = grid_search.best_params_['max_features']
best_rf_model = grid_search.best_estimator_
#access best params
print("Best number of trees:", best_n_estimators)
print("Best max features:", best_max_features)
print("Best model", best_rf_model)


#actually train model and predict
rf2 = RandomForestClassifier(n_estimators = best_n_estimators, max_features = best_max_features).fit(doc_term_matrix_array_train, df_train['Label'])
'''
y_pred = rf2.predict(doc_term_matrix_array_test)
print(accuracy_score(df_test['Label'], y_pred))
'''

# Print training and validation accuracy for each fold
print("Training and Validation Accuracy for Each Fold:")
for i, (train_res, test_res) in enumerate(zip(history.cv_results_, history.cv_results_)):
    print(f"Fold {i+1}: Training results: {train_res}, Validation Results: {test_res}")

