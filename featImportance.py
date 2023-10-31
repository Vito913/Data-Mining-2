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
from sklearn.feature_selection import chi2, SelectKBest, mutual_info_classif
from sklearn.inspection import permutation_importance
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

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
print("train df shape", docTermMatrixTrain.shape)
print("test df shape", docTermMatrixTest.shape)

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


# Set up the KFold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
'''

########### CLASSIFICATION TREES #####################
# Define values to experiment with (alpha for tree)
param_grid_tree = {"ccp_alpha": np.linspace(0, 0.2, 20)}
clf = tree.DecisionTreeClassifier()
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid_tree, cv=kf, n_jobs=-1)

#search for best params config using 10-fold cross validation
#for every parameters combo, model is trained (on 9 folds) and tested (on 1 fold) 10 times; val. accuracy is mean of all 10 trials per param combo!
grid_search.fit(x, y)
best_alpha= grid_search.best_params_['ccp_alpha']
best_tree_model = grid_search.best_estimator_
best_accuracy = grid_search.best_score_

#Train and test on the test set Random Forest
tree2 = tree.DecisionTreeClassifier(ccp_alpha=best_alpha ).fit(doc_term_matrix_array_train, df_train['Label'])
train_accuracy = tree2.score(doc_term_matrix_array_train, df_train['Label'])
print("best params train accuracy TREE", train_accuracy)

########### FEATURE ANALISYS #####################
'''

################ RANDOM FOREST #####################
'''
param_grid = {
    #'n_estimators': [200, 300, 400, 500, 600,700, 800, 900],  # List of different numbers of trees
    'n_estimators': [2],
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
print("time for greed search", end_time - start_time)

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
'''
################################# FEATURE ANALYSIS ###########################
'''
##TREE AND RANDOM FOREST
treeLike_models = [tree2, rf2]
N = 5
for i, model in enumerate(treeLike_models):
    #inspect features (default)
    print("Working on ", str(model), "\n")
    
    feat_importances = pd.Series(model.feature_importances_, index = docTermMatrixTrain.columns).sort_values(ascending = True)
    print("feat importances", feat_importances)
    top_n_features_test = feat_importances.nlargest(5)
    print(top_n_features_test)
    top_n_features_test.plot(kind = 'barh')
    plt.savefig(str(model) + "_default_topNfeats.jpg")

    
    # Calculate permutation importance for class 1
    result_class_1_train = permutation_importance(model, doc_term_matrix_array_train, (df_train["Label"] == 1).astype(int),
                                                n_repeats=2, random_state=42, n_jobs=2)

    # Calculate permutation importance for class 0
    result_class_0_train = permutation_importance(model, doc_term_matrix_array_train, (df_train["Label"] == 0).astype(int),
                                                n_repeats=2, random_state=42, n_jobs=2)


    # Use the same indices to sort features for both class 1 and class 0
    sorted_importances_idx_class1 = result_class_1_train.importances_mean.argsort()
    sorted_importances_idx_class0 = result_class_0_train.importances_mean.argsort()
    # Extract feature names based on the sorted indices
    feature_names_sorted_class1 = docTermMatrixTrain.columns[sorted_importances_idx_class1]
    feature_names_sorted_class0 = docTermMatrixTrain.columns[sorted_importances_idx_class0]

    # Get permutation importances for sorted features for class 1 and class 0
    importances_class_1_train = result_class_1_train.importances[sorted_importances_idx_class1].T
    importances_class_0_train = result_class_0_train.importances[sorted_importances_idx_class0].T

    # Create DataFrames for visualization
    top_n_features_class_1_train = pd.DataFrame(importances_class_1_train, columns=feature_names_sorted_class1)
    top_n_features_class_0_train = pd.DataFrame(importances_class_0_train, columns=feature_names_sorted_class0)
     
    print(top_n_features_class_1_train)
    print(top_n_features_class_0_train)

    top_n_features_class_1_train.plot(kind='barh')
    plt.title("Top {} Features for Class 1 (Permutation Importance)".format(N))
    plt.xlabel("Decrease in Accuracy Score")
    plt.savefig(str(model) + "_TopNFeats_Class1.jpg")
    plt.close()

    # Plot and save top N features for class 0 using permutation importance
    top_n_features_class_0_train.plot(kind='barh')
    plt.title("Top {} Features for Class 0 (Permutation Importance)".format(N))
    plt.xlabel("Decrease in Accuracy Score")
    plt.savefig(str(model) + "_TopNFeats_Class0.jpg")
    plt.close()
    '''





######################## LOGISTIC REGRESSION ############################
lambda_values = {'C': [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]}
logistic_regression = LogisticRegression(penalty='l1', solver='liblinear')

grid_search = GridSearchCV(estimator=logistic_regression, param_grid=lambda_values, cv=kf)

grid_search.fit(doc_term_matrix_array_train, df_train['Label'])
# Get the best hyperparameter
best_lambda = grid_search.best_params_['C']

# Get the best cross-validated score
best_accuracy = grid_search.best_score_


logreg2 = LogisticRegression(penalty='l1',  C=best_lambda, solver='liblinear')
logreg2.fit(doc_term_matrix_array_train, df_train['Label'])
y_pred = logreg2.predict(doc_term_matrix_array_test)
test_accuracy = accuracy_score(df_test['Label'], y_pred)
print("best params test accuracy", test_accuracy )


###################### FEATURE IMPORTANCE ANALISYS ###########################

# Get importance
importance = logreg2.coef_[0]

# Get indices of top 5 positive and negative coefficients
top_positive_indices = np.argsort(-importance)[:5]
top_negative_indices = np.argsort(importance)[:5]

# Extract corresponding coefficients and words
top_positive_coeffs = importance[top_positive_indices]
top_negative_coeffs = importance[top_negative_indices]


#REMEMBER TO USE UNI+BIGRAM FEATURE NAMES HERE!! (0 is true, 1 is deceptive)

top_deceptive_words = [feature_names_train[index] for index in top_positive_indices]
top_true_words = [feature_names_train[index] for index in top_negative_indices]

# Concatenate positive and negative coefficients and words
selected_coeffs = np.concatenate((top_positive_coeffs, top_negative_coeffs))
selected_words = top_deceptive_words + top_true_words

# Plot feature importance with corresponding words
plt.figure(figsize=(8, 6))
plt.bar(selected_words, selected_coeffs, color=['red' if x >= 0 else 'green' for x in selected_coeffs])
plt.xlabel('Words')
plt.ylabel('Coefficient Value')
plt.title('Top 5 Class 1 (deceptive)  and Class 0 (true) Coefficients with Corresponding Words')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.savefig("FIA_logregBest.jpg")


################### NAIVE BAYES ###################
k_values = [ 750, 1000]
pipeline = Pipeline([
    ('chi2', SelectKBest(chi2)),
    ('naive_bayes', MultinomialNB())
])

# Create a parameter grid with the k values
param_grid = {
    'chi2__k': k_values
}

# Perform Grid Search with 10-fold cross-validation
grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=10, scoring='accuracy')

# Fit the Grid Search on the data
grid_search.fit(x, y)
finalK = list(grid_search.best_params_.values())[0]

selector = SelectKBest(chi2, k=finalK)
X_train_selected = selector.fit_transform(doc_term_matrix_array_train, df_train['Label'])
selected_feature_indices = selector.get_support(indices=True)
# Get the names of the selected features
selected_feature_names_train = [feature_names_train[index] for index in selected_feature_indices]
print(len(selected_feature_names_train))
X_test_selected = doc_term_matrix_array_test[:, selected_feature_indices]
#Train and test on the test set Naive Bayes
mnb2 = MultinomialNB().fit(X_train_selected , df_train['Label'])

##################### FEATURE ANALISYS (https://stackoverflow.com/questions/50526898/how-to-get-feature-importance-in-naive-bayes)
#normalize=True outputs proportion 
prob_pos = df_train['Label'].value_counts(normalize=True)[0]
prob_neg = df_train['Label'].value_counts(normalize=True)[1]

df_nbf = pd.DataFrame()
df_nbf.index = selected_feature_indices
print(df_nbf.index)
# Convert log probabilities to probabilities. 
df_nbf['deceptive'] = np.e**(mnb2.feature_log_prob_[0, :])
df_nbf['true'] = np.e**(mnb2.feature_log_prob_[1, :])

 
df_nbf['odds_deceptive'] = (df_nbf['deceptive']/df_nbf['true'])*(prob_pos /prob_neg)
df_nbf['odds_true'] = (df_nbf['true']/df_nbf['deceptive'])*(prob_neg/prob_pos )

# Sort the odds ratios for both classes
sorted_positive_indices = df_nbf['odds_deceptive'].sort_values(ascending=False).index[:5]
sorted_negative_indices = df_nbf['odds_true'].sort_values(ascending=False).index[:5]
# Extract top 5 positive and negative odds ratios along with their corresponding words
top_positive_ratios = df_nbf['odds_deceptive'][sorted_positive_indices]
top_negative_ratios = df_nbf['odds_true'][sorted_negative_indices]

# Here are the top5 most important words of your positive class:
odds_pos_top5 = df_nbf.sort_values('odds_deceptive',ascending=False)['odds_deceptive'][:5]
# Here are the top5 most important words of your negative class:
odds_neg_top5 = df_nbf.sort_values('odds_true',ascending=False)['odds_true'][:5]

top_positive_words = [feature_names_train[i] for i in sorted_positive_indices[:5]]
top_negative_words = [feature_names_train[i] for i in sorted_negative_indices[:5]]

# Print the top 5 positive and negative words along with their odds ratios
print("Top 5 deceptive words:")
for word, odds_ratio in zip(top_positive_words, top_positive_ratios):
    print(f"{word}: {odds_ratio}")

print("\nTop 5 true words:")
for word, odds_ratio in zip(top_negative_words, top_negative_ratios):
    print(f"{word}: {odds_ratio}")

# Data for plotting
words = top_positive_words + top_negative_words
odds_ratios = list(top_positive_ratios) + list(top_negative_ratios)
labels = ['Deceptive'] * 5 + ['True'] * 5

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(range(len(words)), odds_ratios, tick_label=words, color=['red', 'red', 'red', 'red', 'red','green', 'green', 'green', 'green', 'green',])
plt.xlabel('Words')
plt.ylabel('Odds Ratios')
plt.title('Top 5 Deceptive and True Words with Odds Ratios')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("FIA_naiveBest.jpg")
