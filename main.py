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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

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
#print("Train Set:")
#print(train_df.head(10),len(train_df))

#print("\nTest Set:")
#print(test_df.head(10),len(test_df))

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
words_only_reviews = [[' '.join(word for word, pos in review)] for review in df_train['Review Text']]
# Flatten the list of words into a single list
all_words = list(chain.from_iterable(words_only_reviews))
# Convert the list of words into space-separated strings for each review
doc_strings = [' '.join(review) for review in words_only_reviews]
# Initialize the CountVectorizer
vectorizer = CountVectorizer()
# Transform the documents into a document-term co-occurrence matrix
doc_term_matrix = vectorizer.fit_transform(doc_strings)
# Convert the document-term matrix to an array
doc_term_matrix_array = doc_term_matrix.toarray()
if drop:
    doc_term_matrix_array = doc_term_matrix_array[:, (doc_term_matrix_array.sum(axis=0) >= drop_percent * doc_term_matrix_array.shape[0])]
# Get the feature names (words) corresponding to the columns of the matrix
feature_names = vectorizer.get_feature_names_out()

## Check which words happen less than drop_percent times and remove them from the matrix
docTermMatrix = pd.DataFrame(doc_term_matrix_array, columns=feature_names, index=filenamesTrain)
#print(docTermMatrix)

# assigning two variables for the datapoints and the labels for cross-validation
x = doc_term_matrix_array
y = df_train['Label']

# split the data into 10 folds and every time train on 9 and test on 1
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Define the range of lambda values to try
Lambda = {'C': [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0]}
best_accuracy = 0
best_lambda = None

for train_index, val_index in kf.split(x):

    X_train, X_val = x[train_index], x[val_index]
    y_train, y_val = y[train_index], y[val_index]

    logistic_regression = LogisticRegression(penalty='l1', solver='liblinear')
    grid_search = GridSearchCV(logistic_regression, Lambda, cv=10)

    # Perform the grid search
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameter
    best_lambda = grid_search.best_params_['C']

    # Train the model with the best lambda
    best_model = LogisticRegression(penalty='l1', solver='liblinear', C=best_lambda)
    best_model.fit(X_train, y_train)

    # Predict on the validation set
    y_pred = best_model.predict(X_val)

    # Evaluate the model (e.g., calculate accuracy)
    accuracy = best_model.score(X_val, y_val)

    # picking the best accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_lambda_final = best_lambda

print(best_accuracy,best_lambda)
#final_logistic_regression = LogisticRegression(penalty='l1', solver='liblinear', C=best_lambda_final)
#final_logistic_regression.fit(x, y)

