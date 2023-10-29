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
from itertools import chain, product
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.pipeline import Pipeline

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')

# Checks if we are dropping a percentage of the terms
drop = True
# The amount of terms to drop
drop_percent = 0.0015
# Checks if we are using part of speech tagging
part_of_speech = True
# Checks if we are using stemming
stemming = True

onlyWords = False

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
    
    # Remove unnecessary whitespaces
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


# Create a list of lists containing the words for each review
words = [[' '.join(word for word, _ in review)] for review in df_train['Review Text']]
words_test = [[' '.join(word for word, _ in review)] for review in df_test['Review Text']]
words_and_pos = [[' '.join(f"{word}_{pos}" for word, pos in review)] for review in df_train['Review Text']]
words_and_pos_test = [[' '.join(f"{word}_{pos}" for word, pos in review)] for review in df_test['Review Text']]

# Create document-term matrix for train data -> rows = files and columns = terms
doc_strings_train = [' '.join(review) for review in words]
doc_strings_test = [' '.join(review) for review in words_test]
doc_strings_train_pos = [' '.join(review) for review in words_and_pos]
doc_strings_test_pos = [' '.join(review) for review in words_and_pos_test]

# Initialize the CountVectorizer
vectorizerTrain = CountVectorizer()
doc_term_matrix_train = vectorizerTrain.fit_transform(doc_strings_train)

vectorizerTest = CountVectorizer(vocabulary=vectorizerTrain.vocabulary_)
doc_term_matrix_test = vectorizerTest.transform(doc_strings_test)

vectorizerTrainPos = CountVectorizer()
doc_term_matrix_train_pos = vectorizerTrainPos.fit_transform(doc_strings_train_pos)

vectorizerTestPos = CountVectorizer(vocabulary=vectorizerTrainPos.vocabulary_)
doc_term_matrix_test_pos = vectorizerTestPos.transform(doc_strings_test_pos)

vectorizedBigramTrain = CountVectorizer(ngram_range=(2, 2))
doc_term_matrix_train_bigram = vectorizedBigramTrain.fit_transform(doc_strings_train)

vectorizedBigramTest = CountVectorizer(vocabulary=vectorizedBigramTrain.vocabulary_)
doc_term_matrix_test_bigram = vectorizedBigramTest.transform(doc_strings_test)


# Convert the document-term matrix to an array
doc_term_matrix_array_train = doc_term_matrix_train.toarray()
doc_term_matrix_array_test = doc_term_matrix_test.toarray()
doc_term_matrix_array_train_pos = doc_term_matrix_train_pos.toarray()
doc_term_matrix_array_test_pos = doc_term_matrix_test_pos.toarray()
doc_term_matrix_array_train_bigram = doc_term_matrix_train_bigram.toarray()
doc_term_matrix_array_test_bigram = doc_term_matrix_test_bigram.toarray()


'''
#to test if words in doc makes sense
# Select the specific row by its index (document name) 
# specific_row = docTermMatrixTest.loc['d_allegro_1.txt']
# Find non-zero elements and their column names for the specified row
non_zero_elements = specific_row[specific_row != 0]
# Print non-zero elements and their column names for the specified row
for column_name, value in non_zero_elements.items():
    print(f'Column Name: {column_name}, Value: {value}')
'''


# assigning two variables for the datapoints and the labels for cross-validation

######################## LOGISTIC REGRESSION ############################

# split the data into 10 folds and every time train on 9 and test on 1

kf = KFold(n_splits=10, shuffle=True, random_state=42)
def logistic_regression_classification(x, y, kf):
    # Define the range of lambda values to try
    Lambda = {'C': [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0]}

    logistic_regression = LogisticRegression(penalty='l1', solver='liblinear')

    grid_search = GridSearchCV(estimator=logistic_regression, param_grid=Lambda, cv=kf)

    # Perform the grid search
    grid_search.fit(x, y)

    # Get the best hyperparameter
    best_lambda = grid_search.best_params_['C']

    # Get the best model
    best_model = grid_search.best_estimator_

    # Get the best cross-validated score
    best_accuracy = grid_search.best_score_

    return best_accuracy, best_lambda

########### CLASSIFICATION TREES #####################

def decision_tree_classification(x, y, kf):
    param_grid = {"ccp_alpha": np.linspace(0, 0.2, 20)}
    best_alpha = 0
    best_accuracy = 0

    for train_index, val_index in kf.split(x):

        X_train, X_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]

        clf = tree.DecisionTreeClassifier()
        grid_search = GridSearchCV(clf, param_grid=param_grid)
        grid_search.fit(X_train, y_train)
        best_clf = grid_search.best_estimator_
        best_clf.fit(X_train, y_train)
        y_pred = best_clf.predict(X_val)
        accuracy = best_clf.score(X_val, y_val)
        print("accuracy:", accuracy)

        # Active selection of best hyperparameters
        param_grid["ccp_alpha"] = np.linspace(max(0, grid_search.best_params_['ccp_alpha']-0.02), 
                                               min(0.1, grid_search.best_params_['ccp_alpha']+0.02), 20)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_alpha = grid_search.best_params_['ccp_alpha']

    return best_accuracy, best_alpha
################ RANDOM FOREST #####################

def random_forest_classification(x: pd.DataFrame, y: pd.DataFrame,kf) -> (float, str, RandomForestClassifier):
    param_grid = {
        'n_estimators': [200, 300, 400, 500, 600,700, 800, 900],  # List of different numbers of trees
        'max_features': ['sqrt', 'log2', None]  # Different options for max_features
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=kf, n_jobs=-1)

    #search for best params config using 10-fold cross validation
    #for every parameters combo, model is trained (on 9 folds) and tested (on 1 fold) 10 times; val. accuracy is mean of all 10 trials per param combo!
    start_time = time.time()
    history = grid_search.fit(x, y)
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

    return best_n_estimators, best_max_features, best_rf_model




best_overall_alpha = 0
best_overall_lambda = 0
best_overall_n_estimators = 0

for drop, drop_percent, part_of_speech, in product([False,True], [0.0015, 0.002, 0.0025, 0.003], [True, False]):
    
    if drop:
        doc_term_matrix_array_train = doc_term_matrix_array_train[:, (doc_term_matrix_array_train.sum(axis=0) >= drop_percent * doc_term_matrix_array_train.shape[0])]
        doc_term_matrix_array_train_pos = doc_term_matrix_array_train_pos[:, (doc_term_matrix_array_train_pos.sum(axis=0) >= drop_percent * doc_term_matrix_array_train_pos.shape[0])]
        # Update the filenamesTrain list to match the number of rows in the modified doc_term_matrix_array_train

    # Get the feature names (words) corresponding to the columns of the matrix
    feature_names_train = vectorizerTrain.get_feature_names_out()
    feature_names_train_pos = vectorizerTrainPos.get_feature_names_out()
    
    # Ensure that the test matrix only includes words from the training matrix
    docTermMatrixTest = pd.DataFrame(doc_term_matrix_array_test, columns=feature_names_train, index=filenamesTest)
    docTermMatrixTestPos = pd.DataFrame(doc_term_matrix_array_test_pos, columns=feature_names_train_pos, index=filenamesTest)
    
    ## Check which words happen less than drop_percent times and remove them from the matrix
    docTermMatrixTrain = pd.DataFrame(doc_term_matrix_array_train, columns=feature_names_train, index=filenamesTrain)
    docTermMatrixTrainPos = pd.DataFrame(doc_term_matrix_array_train_pos, columns=feature_names_train_pos, index=filenamesTrain)

    if part_of_speech:
        doc_term_matrix_array_train = doc_term_matrix_array_train_pos
        doc_term_matrix_array_test = doc_term_matrix_array_test_pos
    else:
        doc_term_matrix_array_train = doc_term_matrix_array_train
        doc_term_matrix_array_test = doc_term_matrix_array_test
    y = df_train['Label']
    x = doc_term_matrix_array_train
    
    accuracy_logistic, current_best_lambda = logistic_regression_classification(x, y, kf)
    print("best_lambda", current_best_lambda)
    print("best_accuracy", accuracy_logistic)
    best_n_estimators, best_max_features, best_rf_model = random_forest_classification(x, y,kf)
    print("best_n_estimators", best_n_estimators)
    print("best_max_features", best_max_features)
    print("best_rf_model", best_rf_model)
    accuracy_trees, best_alpha = decision_tree_classification(x, y, kf)
    print("best_alpha", best_alpha)
    print("accuracy_trees", accuracy_trees)
    if accuracy_logistic > best_overall_lambda:
        best_overall_lambda = accuracy_logistic
        best_overall_alpha = current_best_lambda
    if accuracy_trees > best_overall_alpha:
        best_overall_alpha = accuracy_trees
        best_overall_lambda = current_best_lambda
    if best_n_estimators > best_overall_n_estimators:
        best_overall_n_estimators = best_n_estimators
        best_overall_max_features = best_max_features




################ Naive Bayes without bigram nor features selection "Chi-square" #####################
# Create a pipeline with feature selection and Naive Bayes classifier
pipeline = Pipeline([
    ('chi2', SelectKBest(chi2)),
    ('naive_bayes', MultinomialNB())
])

# Define a range of k values to experiment with
k_values = [ 500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
# Create a parameter grid with the k values
param_grid = {
    'chi2__k': k_values
}

# Perform Grid Search with 10-fold cross-validation
grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=10, scoring='accuracy')

# Fit the Grid Search on the data
grid_search.fit(x, y)

print("CHI SQUARE")
# Print validation accuracy for each parameter combination
print("Validation Accuracy for Each Parameter Combination:")
for params, test_score in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score']):
    print(f"Parameters: {params}, Validation Accuracy: {test_score:.4f}")

# Print the best parameters and corresponding accuracy score
print("Best Parameters: ", grid_search.best_params_)
print("Best Accuracy Score: ", grid_search.best_score_)




#Train and test on the test set Random Forest

rf2 = RandomForestClassifier(n_estimators = best_overall_n_estimators, max_features = best_overall_max_features).fit(doc_term_matrix_array_train, df_train['Label'])
train_accuracy =rf2.score(doc_term_matrix_array_train, df_train['Label'])
print("best params train accuracy", train_accuracy)
#predict on test set
y_pred = rf2.predict(doc_term_matrix_array_test)
test_accuracy = accuracy_score(df_test['Label'], y_pred)
print("best params test accuracy random random forest", test_accuracy )

#Train and test on the test set Logistic Regression 
logreg2= LogisticRegression(penalty='l1', C=best_overall_lambda, solver='liblinear').fit(doc_term_matrix_array_train, df_train['Label'])
train_accuracy_logreg = logreg2.score(doc_term_matrix_array_train, df_train['Label'])
print("best params train accuracy", train_accuracy_logreg)
#predict on test set
y_pred_logreg = logreg2.predict(doc_term_matrix_array_test)
test_accuracy_logreg= accuracy_score(df_test['Label'], y_pred_logreg)
print("best params test accuracy logreg", test_accuracy_logreg)

