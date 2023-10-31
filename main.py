import os
from matplotlib import pyplot as plt
import pandas as pd
import string
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
import seaborn as sns

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

def evaluate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    acc = (cm[0, 0] + cm[1, 1]) / np.sum(cm)
    prec = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    rec = cm[0, 0] / (cm[0, 0] + cm[1, 0])
    return cm, acc, prec, rec



'''
# Checks if we are dropping a percentage of the terms
drop = True
# The amount of terms to drop
drop_percent = 0.0015
# Checks if we are using part of speech tagging
part_of_speech = True
# Checks if we are using stemming
stemming = True
onlyWords = False
'''

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

# Initialize the CountVectorizer
vectorizerTrain = CountVectorizer()
vectorizerTrainPos = CountVectorizer()
vectorizedBigramTrain = CountVectorizer(ngram_range=(1, 2))
vectorizedBigramTrainPos = CountVectorizer(ngram_range=(1, 2))
# Create document-term matrix for train data -> rows = files and columns = terms
doc_term_matrix_train = vectorizerTrain.fit_transform(doc_strings_train)
doc_term_matrix_train_pos = vectorizerTrainPos.fit_transform(doc_strings_train_pos)
doc_term_matrix_train_bigram = vectorizedBigramTrain.fit_transform(doc_strings_train)
doc_term_matrix_train_bigram_pos = vectorizedBigramTrainPos.fit_transform(doc_strings_train_pos)

# Create document-term matrix for test data -> rows = files and columns = terms
doc_term_matrix_test = vectorizerTrain.transform(doc_strings_test)
doc_term_matrix_test_pos = vectorizerTrainPos.transform(doc_strings_test_pos)
doc_term_matrix_test_bigram = vectorizedBigramTrain.transform(doc_strings_test)
doc_term_matrix_test_bigram_pos = vectorizedBigramTrainPos.transform(doc_strings_test_pos)

# Convert the document-term matrix to an array
doc_term_matrix_array_train = doc_term_matrix_train.toarray()
doc_term_matrix_array_test = doc_term_matrix_test.toarray()
doc_term_matrix_array_train_pos = doc_term_matrix_train_pos.toarray()
doc_term_matrix_array_test_pos = doc_term_matrix_test_pos.toarray()
doc_term_matrix_array_train_bigram = doc_term_matrix_train_bigram.toarray()
doc_term_matrix_array_test_bigram = doc_term_matrix_test_bigram.toarray()
doc_term_matrix_array_train_bigram_pos = doc_term_matrix_train_bigram_pos.toarray()
doc_term_matrix_array_test_bigram_pos = doc_term_matrix_test_bigram_pos.toarray()


######################## LOGISTIC REGRESSION ############################

def logistic_regression_classification(x, y, kf, lambda_values):
    # Define the range of lambda values to try

    logistic_regression = LogisticRegression(penalty='l1', solver='liblinear')

    grid_search = GridSearchCV(estimator=logistic_regression, param_grid=lambda_values, cv=kf)

    # Perform the grid search
    grid_search.fit(x, y)

    # Get the best hyperparameter
    best_lambda = grid_search.best_params_['C']

    # Get the best cross-validated score
    best_accuracy = grid_search.best_score_

    return best_accuracy, best_lambda

########### CLASSIFICATION TREES #####################

def decision_tree_classification(x, y, kf, param_grid):
    clf = tree.DecisionTreeClassifier()
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=kf, n_jobs=-1)

    #search for best params config using 10-fold cross validation
    #for every parameters combo, model is trained (on 9 folds) and tested (on 1 fold) 10 times; val. accuracy is mean of all 10 trials per param combo!
    grid_search.fit(x, y)
    best_alpha= grid_search.best_params_['ccp_alpha']
    best_tree_model = grid_search.best_estimator_
    best_accuracy = grid_search.best_score_

    return best_accuracy, best_alpha

################ RANDOM FOREST #####################

def random_forest_classification(x: pd.DataFrame, y: pd.DataFrame,kf: KFold, param_grid: dict) -> (float, str, RandomForestClassifier):
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=kf, n_jobs=-1)

    #search for best params config using 10-fold cross validation
    #for every parameters combo, model is trained (on 9 folds) and tested (on 1 fold) 10 times; val. accuracy is mean of all 10 trials per param combo!
    grid_search.fit(x, y)
    #go on with the best config!
    best_n_estimators = grid_search.best_params_['n_estimators']
    best_max_features = grid_search.best_params_['max_features']
    best_accuracy = grid_search.best_score_
    return  best_accuracy, best_n_estimators, best_max_features


################ Naive Bayes without bigram nor features selection "Chi-square" #####################
def naive_bayes(x, y, k_values):
    # Create a pipeline with feature selection and Naive Bayes classifier
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
    
    # Store mean accuracy values and corresponding parameter values    
    # mean_accuracies = []     
    # k_parameters = []     
    # for params, mean_score, _ in grid_search.cv_results_:         
    #     mean_accuracies.append(mean_score)         
    #     k_parameters.append(params['chi2__k'])     
    #     # Plot mean accuracy vs lambda values    

    
    
    return grid_search.best_score_, finalK,# mean_accuracies, k_parameters




## Setting up the variables for the best model among all configurations (i.e. drop y/n, unigram/bigram, pos..)
best_overall_alpha = 0.0
best_overall_lambda = 0.0
best_overall_n_estimators = 0.0
best_overall_naive = 0

##Setting up the accuracies for the best model among all configurations (i.e. drop y/n, unigram/bigram, pos..)
acc_logistic = 0
acc_rf = 0
acc_tree = 0
acc_naive = 0

#SEARCH PARAMS
# # Define values to experiment with (labda for logreg)
# lambda_values = {'C': [1000.0, 2000.0]}
# # Define values to experiment with (n of trees and max features for rf)
# param_grid_forest = {
#     'n_estimators': [5, 10],  # List of different numbers of trees
#     'max_features': ['sqrt']  # Different options for max_features
# }
# # Define values to experiment with (alpha for tree)
# param_grid_tree = {"ccp_alpha": np.linspace(0, 0.2, 1)}
# # Define values to experiment with (k feature selection for naive bayes)
# k_values = [1000, 1200]
# # Define values to experiment with (labda for logreg)

lambda_values = {'C': [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]}
# Define values to experiment with (n of trees and max features for rf)
param_grid_forest = {
    'n_estimators': [400, 500, 600, 700, 800],  # List of different numbers of trees
    'max_features': ['sqrt', 'log2', None]  # Different options for max_features
}
# Define values to experiment with (alpha for tree)
param_grid_tree = {"ccp_alpha": np.linspace(0, 0.2, 20)}
# Define values to experiment with (k feature selection for naive bayes)
k_values = [ 500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]


# Set up the KFold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Creating a map that will be returned with the best models hyperparameters
best_tree_model = {"drop_percent": 0, "part_of_speech": True, "best_alpha": 0, "best_accuracy": 0, "bigram": False, "bigram_pos":False}
best_logistic_model = {"drop_percent": 0, "part_of_speech": True, "best_lambda": 0, "best_accuracy": 0,"bigram": False, "bigram_pos":False}
best_random_forest = {"drop_percent": 0, "part_of_speech": True, "best_n_estimator": 0, "best_accuracy": 0, "best_max_features": 0,"bigram": False, "bigram_pos":False}
best_naive = {"drop_percent": 0, "part_of_speech": True, "best_k": 0, "best_accuracy": 0, "bigram": False,"bigram_pos":False }

for bigram, pos_bigram in product([False, True],[False, True]):
    print("bigram", bigram)
    #for drop, drop_percent, part_of_speech, in product([False,True], [0.0015, 0.002, 0.0025, 0.003], [False,True]):
    for drop_percent, part_of_speech in product([0.0 ,0.0012, 0.0015 , 0.002, 0.0025], [False,True]):
        # i = 0
        # i+=1

        # Checks if part of speech tagging is used
        if part_of_speech:
            current_matrix_train = doc_term_matrix_array_train_pos
            current_matrix_test = doc_term_matrix_array_test_pos     
        elif bigram:
            current_matrix_train = doc_term_matrix_array_train_bigram
            current_matrix_test = doc_term_matrix_array_test_bigram
        elif pos_bigram:
            current_matrix_train = doc_term_matrix_array_train_bigram_pos
            current_matrix_test = doc_term_matrix_array_test_bigram_pos
        else:
            current_matrix_train = doc_term_matrix_array_train
            current_matrix_test = doc_term_matrix_array_test
        
        if part_of_speech:
            feature_names_train = vectorizerTrainPos.get_feature_names_out()
        elif bigram:
            feature_names_train = vectorizedBigramTrain.get_feature_names_out()
        elif pos_bigram:
            feature_names_train = vectorizedBigramTrainPos.get_feature_names_out()
        else:
            feature_names_train = vectorizerTrain.get_feature_names_out()

        if bigram or pos_bigram:
            used_drop = drop_percent * 5
        else:
            used_drop = drop_percent

        selected_columns = np.where(current_matrix_train.sum(axis=0) >= used_drop* current_matrix_train.shape[0])[0]
        #current_matrix_train = current_matrix_train[:, selected_columns]
        current_matrix_train = current_matrix_train[:, (current_matrix_train.sum(axis=0) >= used_drop * current_matrix_train.shape[0])]
        feature_names_train = [feature_names_train[idx] for idx in selected_columns]  


        ## Check which words happen less than drop_percent times and remove them from the matrix
        docTermMatrixTrain = pd.DataFrame(current_matrix_train, columns=feature_names_train, index=filenamesTrain)
        # Set the labels
        x = doc_term_matrix_array_train
        y = df_train['Label']
        
        #DO CV WITH ALL 4 MODELS BY CALLING THEIR FUNCTIONS
        accuracy_logistic, current_best_lambda = logistic_regression_classification(x, y, kf,lambda_values)
        print("logistic regression done")
        accuracy_forest, best_n_estimators, best_max_features = random_forest_classification(x, y,kf, param_grid_forest)
        print("random forest done")
        accuracy_trees, best_alpha = decision_tree_classification(x, y, kf, param_grid_tree)
        print("single tree done")
        best_accuracy_naive, best_k = naive_bayes(x, y, k_values)
        print("naive bayes done")
        #update best overall params and fill in dictionary with current parameters
        #LOGREG
        if accuracy_logistic > acc_logistic:
            acc_logistic = accuracy_logistic
            best_overall_lambda = current_best_lambda
            best_logistic_model["bigram"] = bigram
            best_logistic_model["drop_percent"] = drop_percent
            best_logistic_model["part_of_speech"] = part_of_speech
            best_logistic_model["best_lambda"] = current_best_lambda
            best_logistic_model["best_accuracy"] = accuracy_logistic
            best_logistic_model["bigram_pos"] = pos_bigram
        #TREE
        if accuracy_trees > acc_tree:
            acc_tree = accuracy_trees
            best_overall_alpha = best_alpha
            best_tree_model["bigram"] = bigram
            best_tree_model["drop_percent"] = drop_percent
            best_tree_model["part_of_speech"] = part_of_speech
            best_tree_model["best_alpha"] = best_overall_alpha
            best_tree_model["best_accuracy"] = acc_tree
            best_tree_model["bigram_pos"] = pos_bigram

        #FOREST
        if accuracy_forest > acc_rf:
            acc_rf = accuracy_forest
            best_overall_n_estimators = best_n_estimators
            best_overall_max_features = best_max_features
            best_random_forest["bigram"] = bigram
            best_random_forest["drop_percent"] = drop_percent
            best_random_forest["part_of_speech"] = part_of_speech
            best_random_forest["best_accuracy"] = accuracy_forest
            best_random_forest["best_max_features"] = best_max_features
            best_random_forest["best_n_estimator"] = best_n_estimators
            best_random_forest["bigram_pos"] = pos_bigram

        #NAIVE BAYES
        if best_accuracy_naive > acc_naive:
            acc_naive = best_accuracy_naive
            best_naive["bigram"] = bigram
            best_naive["drop_percent"] = drop_percent
            best_naive["part_of_speech"] = part_of_speech
            best_naive["best_k"] = best_k
            best_naive["best_accuracy"] = best_accuracy_naive
            best_naive["bigram_pos"] = pos_bigram
            #best_naive["mean_acc"] = mean_acc_bayes
            #best_naive["k_params"] = k_params
        
    if bigram or pos_bigram:
        best_naive_bigram = best_naive
        best_logistic_model_bigram = best_logistic_model
        best_random_forest_bigram = best_random_forest
        best_tree_model_bigram = best_tree_model
    else:
        best_naive_unigrams = best_naive
        best_logistic_model_unigrams = best_logistic_model
        best_random_forest_unigrams = best_random_forest
        best_tree_model_unigrams = best_tree_model
        
        
                    
    # Print after being done
    print("decision tree results", best_tree_model)
    print("logistic regression results", best_logistic_model)
    print("random forest results", best_random_forest)
    print("naive bayes results", best_naive)






#PRINT BEST DICTIONARY (best param config) PER MODEL

# plt.figure(figsize=(6, 6))
# plt.plot(best_naive["k_params"], best_naive["mean_acc"], marker= "o")
# plt.xlabel("k")
# plt.ylabel("Accuracy")
# plt.title("Naive Bayes")
# plt.savefig('naive_bayes.png')  # Save the image

#TRAIN AND TEST BEST MODELS AFTER CV

## save the best models dictioneries to a txt file
with open("best_models.txt", "w") as f:
    f.write("best naive unigrams " + str(best_naive_unigrams) + "\n")
    f.write("best naive bigram " + str(best_naive_bigram) + "\n")
    f.write("best logistic unigrams " + str(best_logistic_model_unigrams) + "\n")
    f.write("best logistic bigram " + str(best_logistic_model_bigram) + "\n")
    f.write("best random forest unigrams " + str(best_random_forest_unigrams) + "\n")
    f.write("best random forest bigram " + str(best_random_forest_bigram) + "\n")
    f.write("best tree unigrams " + str(best_tree_model_unigrams) + "\n")
    f.write("best tree bigram " + str(best_tree_model_bigram) + "\n")


################# UNIGRAM STUFF #####################

#Train and test on the test set tree
tree2 = tree.DecisionTreeClassifier(ccp_alpha=best_tree_model_unigrams["best_alpha"] ).fit(doc_term_matrix_array_train, df_train['Label'])
train_accuracy = tree2.score(doc_term_matrix_array_train, df_train['Label'])
print("best params train accuracy TREE", train_accuracy)
#predict on test set
y_pred = tree2.predict(doc_term_matrix_array_test)
evaluate_tree = evaluate(df_test['Label'], y_pred)
print("confusion matrix tree", evaluate_tree[0])

test_accuracy = accuracy_score(df_test['Label'], y_pred)
print("best params test accuracy TREE", test_accuracy )


plt.figure(figsize=(6, 6))
sns.set(font_scale=1.2)
sns.heatmap(evaluate_tree[0], annot=True, fmt='g', cmap='Blues',
            xticklabels=['Class 0', 'Class 1'],  # type: ignore
            yticklabels=['Class 0', 'Class 1']) # type: ignore
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Decision Tree')
plt.tight_layout()
plt.savefig('confused_decision_unigram.png')  # Save the image
# plt.show()

#Train and test on the test set Random Forest
rf2 = RandomForestClassifier(n_estimators = best_random_forest_unigrams["best_n_estimator"] , max_features = best_random_forest_unigrams["best_max_features"]).fit(doc_term_matrix_array_train, df_train['Label'])
train_accuracy =rf2.score(doc_term_matrix_array_train, df_train['Label'])
print("best params train accuracy RANDOM FOREST", train_accuracy)
#predict on test set
y_pred = rf2.predict(doc_term_matrix_array_test)
test_accuracy = accuracy_score(df_test['Label'], y_pred)

evaluate_forest = evaluate(df_test['Label'], y_pred)
print("confusion matrix forest", evaluate_forest[0])

print("best params test accuracy RANDOM FOREST", test_accuracy )

plt.figure(figsize=(6, 6))
sns.set(font_scale=1.2)
sns.heatmap(evaluate_forest[0], annot=True, fmt='g', cmap='Blues',
            yticklabels=['Class 0', 'Class 1'],  # type: ignore
            xticklabels=['Class 0', 'Class 1']) # type: ignore
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Random Forest')
plt.tight_layout()
plt.savefig('confused_forest_unigram.png')  # Save the image
# plt.show()

#Train and test on the test set Logistic Regression 
logreg2= LogisticRegression(penalty='l1', C= best_logistic_model_unigrams["best_lambda"] , solver='liblinear').fit(doc_term_matrix_array_train, df_train['Label'])
train_accuracy_logreg = logreg2.score(doc_term_matrix_array_train, df_train['Label'])
print("best params train accuracy LOGREG", train_accuracy_logreg)
#predict on test set
y_pred_logreg = logreg2.predict(doc_term_matrix_array_test)
test_accuracy_logreg= accuracy_score(df_test['Label'], y_pred_logreg)
print("best params test accuracy LOGREG", test_accuracy_logreg)
evaluate_logreg = evaluate(df_test['Label'], y_pred_logreg)
print("confusion matrix logreg", evaluate_logreg[0])

plt.figure(figsize=(6, 6))
sns.set(font_scale=1.2)
sns.heatmap(evaluate_logreg[0], annot=True, fmt='g', cmap='Blues',
            yticklabels=['Class 0', 'Class 1'],  # type: ignore
            xticklabels=['Class 0', 'Class 1']) # type: ignore
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Logistic Regression')
plt.tight_layout()
plt.savefig('confused_logreg_unigram.png')  # Save the image
# plt.show()


#modify train and test matrices based on k features selected during crossvalidation
selector = SelectKBest(chi2, k=best_naive_unigrams["best_k"])
X_train_selected = selector.fit_transform(doc_term_matrix_array_train, df_train['Label'])
selected_feature_indices = selector.get_support(indices=True)
X_test_selected = doc_term_matrix_array_test[:, selected_feature_indices]
#Train and test on the test set Naive Bayes
mnb2 = MultinomialNB().fit(X_train_selected , df_train['Label'])
train_accuracy_mnb2= mnb2.score(X_train_selected , df_train['Label'])
print("best params train accuracy NAIVE BAYES", train_accuracy_mnb2)
#predict on test set
y_pred_mnb2 = mnb2.predict(X_test_selected)
test_accuracy_mnb2 = accuracy_score(df_test['Label'], y_pred_mnb2)
print("best params test accuracy NAIVE BAYES", test_accuracy_mnb2)
evaluate_mnb2 = evaluate(df_test['Label'], y_pred_mnb2)
print("confusion matrix mnb", evaluate_mnb2[0])

plt.figure(figsize=(6, 6))
sns.set(font_scale=1.2)
sns.heatmap(evaluate_mnb2[0], annot=True, fmt='g', cmap='Blues',
            yticklabels=['Class 0', 'Class 1'],  # type: ignore
            xticklabels=['Class 0', 'Class 1']) # type: ignore
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix MNB')
plt.tight_layout()
plt.savefig('confused_mnb2_unigram.png')  # Save the image
# plt.show()


################# BIGRAM STUFF #####################


#Train and test on the test set Random Forest
tree2 = tree.DecisionTreeClassifier(ccp_alpha=best_tree_model_bigram["best_alpha"] ).fit(doc_term_matrix_array_train, df_train['Label'])
train_accuracy = tree2.score(doc_term_matrix_array_train, df_train['Label'])
print("best params train accuracy TREE", train_accuracy)
#predict on test set
y_pred = tree2.predict(doc_term_matrix_array_test)
evaluate_tree = evaluate(df_test['Label'], y_pred)
print("confusion matrix tree", evaluate_tree[0])

test_accuracy = accuracy_score(df_test['Label'], y_pred)
print("best params test accuracy TREE", test_accuracy )


plt.figure(figsize=(6, 6))
sns.set(font_scale=1.2)
sns.heatmap(evaluate_tree[0], annot=True, fmt='g', cmap='Blues',
            xticklabels=['Class 0', 'Class 1'],  # type: ignore
            yticklabels=['Class 0', 'Class 1']) # type: ignore
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Decision Tree')
plt.tight_layout()
plt.savefig('confused_decision_bigram.png')  # Save the image
# plt.show()

#Train and test on the test set Random Forest
rf2 = RandomForestClassifier(n_estimators = best_random_forest_bigram["best_n_estimator"] , max_features = best_random_forest_bigram["best_max_features"]).fit(doc_term_matrix_array_train, df_train['Label'])
train_accuracy =rf2.score(doc_term_matrix_array_train, df_train['Label'])
print("best params train accuracy RANDOM FOREST", train_accuracy)
#predict on test set
y_pred = rf2.predict(doc_term_matrix_array_test)
test_accuracy = accuracy_score(df_test['Label'], y_pred)

evaluate_forest = evaluate(df_test['Label'], y_pred)
print("confusion matrix forest", evaluate_forest[0])

print("best params test accuracy RANDOM FOREST", test_accuracy )

plt.figure(figsize=(6, 6))
sns.set(font_scale=1.2)
sns.heatmap(evaluate_forest[0], annot=True, fmt='g', cmap='Blues',
            yticklabels=['Class 0', 'Class 1'],  # type: ignore
            xticklabels=['Class 0', 'Class 1']) # type: ignore
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Random Forest')
plt.tight_layout()
plt.savefig('confused_forest_bigram.png')  # Save the image
# plt.show()

#Train and test on the test set Logistic Regression 
logreg2= LogisticRegression(penalty='l1', C= best_logistic_model_bigram["best_lambda"] , solver='liblinear').fit(doc_term_matrix_array_train, df_train['Label'])
train_accuracy_logreg = logreg2.score(doc_term_matrix_array_train, df_train['Label'])
print("best params train accuracy LOGREG", train_accuracy_logreg)
#predict on test set
y_pred_logreg = logreg2.predict(doc_term_matrix_array_test)
test_accuracy_logreg= accuracy_score(df_test['Label'], y_pred_logreg)
print("best params test accuracy LOGREG", test_accuracy_logreg)
evaluate_logreg = evaluate(df_test['Label'], y_pred_logreg)
print("confusion matrix logreg", evaluate_logreg[0])

plt.figure(figsize=(6, 6))
sns.set(font_scale=1.2)
sns.heatmap(evaluate_logreg[0], annot=True, fmt='g', cmap='Blues',
            yticklabels=['Class 0', 'Class 1'],  # type: ignore
            xticklabels=['Class 0', 'Class 1']) # type: ignore
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Logistic Regression')
plt.tight_layout()
plt.savefig('confused_logreg_bigram.png')  # Save the image
# plt.show()


#modify train and test matrices based on k features selected during crossvalidation
selector = SelectKBest(chi2, k=best_naive_bigram["best_k"])
X_train_selected = selector.fit_transform(doc_term_matrix_array_train, df_train['Label'])
selected_feature_indices = selector.get_support(indices=True)
X_test_selected = doc_term_matrix_array_test[:, selected_feature_indices]
#Train and test on the test set Naive Bayes
mnb2 = MultinomialNB().fit(X_train_selected , df_train['Label'])
train_accuracy_mnb2= mnb2.score(X_train_selected , df_train['Label'])
print("best params train accuracy NAIVE BAYES", train_accuracy_mnb2)
#predict on test set
y_pred_mnb2 = mnb2.predict(X_test_selected)
test_accuracy_mnb2 = accuracy_score(df_test['Label'], y_pred_mnb2)
print("best params test accuracy NAIVE BAYES", test_accuracy_mnb2)
evaluate_mnb2 = evaluate(df_test['Label'], y_pred_mnb2)
print("confusion matrix mnb", evaluate_mnb2[0])

plt.figure(figsize=(6, 6))
sns.set(font_scale=1.2)
sns.heatmap(evaluate_mnb2[0], annot=True, fmt='g', cmap='Blues',
            yticklabels=['Class 0', 'Class 1'],  # type: ignore
            xticklabels=['Class 0', 'Class 1']) # type: ignore
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix MNB')
plt.tight_layout()
plt.savefig('confused_mnb2_bigram.png')  # Save the image
# plt.show()


