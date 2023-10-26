import os
import pandas as pd
import string
import time
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from itertools import chain
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import numpy as np

#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')

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

############### Data Preprocessing #####################

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


## Part of speech tagging

def pos_tagging(row):
    words = word_tokenize(row)
    return nltk.pos_tag(words)

################# Data Processing  #####################

# Apply the remove_unnecessary function to each row in the df_train and df_test dataframes
df_train['Review Text'] = df_train['Review Text'].apply(remove_unnecessary).apply(stemming).apply(pos_tagging)
df_test['Review Text'] = df_test['Review Text'].apply(remove_unnecessary).apply(stemming).apply(pos_tagging)

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
# Transform the documents into a document-term co-occurrence matrix
doc_term_matrix_train = vectorizerTrain.fit_transform(doc_strings_train)
# Initialize the CountVectorizer with the vocabulary from the training data
vectorizerTest = CountVectorizer(vocabulary=vectorizerTrain.vocabulary_)
# Transform the test documents into a document-term co-occurrence matrix using the same vocabulary
doc_term_matrix_test = vectorizerTest.transform(doc_strings_test)

# Convert the document-term matrix to an array
doc_term_matrix_array_train = doc_term_matrix_train.toarray()
doc_term_matrix_array_test = doc_term_matrix_test.toarray()
if drop:
    doc_term_matrix_array_train = doc_term_matrix_array_train[:, (doc_term_matrix_array_train.sum(axis=0) >= drop_percent * doc_term_matrix_array_train.shape[0])]

# Get the feature names (words) corresponding to the columns of the matrix
feature_names_train = vectorizerTrain.get_feature_names_out()

## Check which words happen less than drop_percent times and remove them from the matrix
docTermMatrixTrain = pd.DataFrame(doc_term_matrix_array_train, columns=feature_names_train, index=filenamesTrain)
# Ensure that the test matrix only includes words from the training matrix
docTermMatrixTest = pd.DataFrame(doc_term_matrix_array_test, columns=feature_names_train, index=filenamesTest)
print(docTermMatrixTrain)
print(docTermMatrixTest)

'''
#to test if words in doc makes sense
# Select the specific row by its index (document name)
specific_row = docTermMatrixTest.loc['d_allegro_1.txt']
# Find non-zero elements and their column names for the specified row
non_zero_elements = specific_row[specific_row != 0]
# Print non-zero elements and their column names for the specified row
for column_name, value in non_zero_elements.items():
    print(f'Column Name: {column_name}, Value: {value}')
'''


# assigning two variables for the datapoints and the labels for cross-validation
x = doc_term_matrix_array_train
y = df_train['Label']


######################## LOGISTIC REGRESSION ############################

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


########### CLASSIFICATION TREES #####################

param_grid = {"ccp_alpha": np.linspace(0, 0.1, 11)}
best_alpha = 0
best_accuracy_2_electric_boogaloo = 0

for train_index, val_index in kf.split(x):

    X_train, X_val = x[train_index], x[val_index]
    y_train, y_val = y[train_index], y[val_index]

    clf = tree.DecisionTreeClassifier()
    grid_search = GridSearchCV(clf, param_grid=param_grid)
    grid_search.fit(X_train, y_train)
    #print("best alpha:", grid_search.best_params_['ccp_alpha'])
    best_clf = grid_search.best_estimator_
    best_clf.fit(X_train, y_train)
    y_pred = best_clf.predict(X_val)
    accuracy = best_clf.score(X_val, y_val)
    print("accuracy:", accuracy)
    
    # Active selection of best hyperparameters
    param_grid["ccp_alpha"] = np.linspace(max(0, grid_search.best_params_['ccp_alpha']-0.01), 
                                           min(0.1, grid_search.best_params_['ccp_alpha']+0.01), 11)

    if accuracy > best_accuracy_2_electric_boogaloo:
        best_accuracy_2_electric_boogaloo = accuracy
        best_alpha = grid_search.best_params_['ccp_alpha']
        
print(best_accuracy_2_electric_boogaloo, best_alpha)


################ RANDOM FOREST #####################


param_grid = {
    'n_estimators': [200, 300, 400, 500, 600,700, 800, 900],  # List of different numbers of trees
    'max_features': ['sqrt', 'log2', None]  # Different options for max_features
    
}
cv = KFold(n_splits=10, shuffle=True, random_state=42) 
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, n_jobs=-1)
#serch for best params config using 10-fold cross validation
#for every parameters combo, model is trained (on 9 folds) and tested (on 1 fold) 10 times; val. accuracy is mean of all 10 trials per param combo!
start_time = time.time()
history = grid_search.fit(doc_term_matrix_array_train, df_train['Label'])
end_time = time.time()
print("time for greed search", start_time - end_time)

# Print validation accuracy for each parameter combination
print("Validation Accuracy for Each Parameter Combination:")
for params, test_score in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score']):
    print(f"Parameters: {params}, Validation Accuracy: {test_score:.4f}")


#go on with the best config!
best_n_estimators = grid_search.best_params_['n_estimators']
best_max_features = grid_search.best_params_['max_features']
best_rf_model = grid_search.best_estimator_
#access best params
print("Best number of trees:", best_n_estimators)
print("Best max features:", best_max_features)
print("Best model", best_rf_model)


#actually train model and predict
rf2 = RandomForestClassifier(n_estimators = best_n_estimators, max_features = best_max_features).fit(doc_term_matrix_array_train, df_train['Label'])
train_accuracy =rf2.score(doc_term_matrix_array_train, df_train['Label'])
print("best params train accuracy", train_accuracy)
#predict on test set
y_pred = rf2.predict(doc_term_matrix_array_test)
test_accuracy = accuracy_score(df_test['Label'], y_pred)
print("best params test accuracy", test_accuracy )