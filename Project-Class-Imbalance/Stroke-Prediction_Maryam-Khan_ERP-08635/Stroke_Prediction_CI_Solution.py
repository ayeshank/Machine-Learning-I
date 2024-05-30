#!/usr/bin/env python
# coding: utf-8

# In[15]:


pip install pandas matplotlib


# In[2]:


import pandas as pd


# In[3]:


import numpy as np


# # Data loading

# In[18]:



file_path = r'C:\Users\Aman ur Rehman\Documents\dataset.csv'

data = pd.read_csv(file_path)


# In[19]:


print(data.head)


# In[20]:



print("\nDataset Statistics:")
print(data.describe())


# # checking class imbalance

# In[21]:



print("\nClass Distribution:")
print(data['stroke'].value_counts())


# In[22]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[23]:




counts = data['stroke'].value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=counts.index, y=counts.values, palette='viridis') 
plt.title('Class Distribution')
plt.ylabel('Frequency')
plt.xlabel('Class (0: no stroke, 1: stroke)')
plt.xticks(range(2), ['no stroke', 'stroke'])
plt.show()


# # Correlation matrix

# In[24]:



corr = data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap='viridis', annot=False)  
plt.title('Correlation Matrix')
plt.show()


# In[25]:


# Calculate correlation matrix
corr = data.corr()

# Calculate correlation matrix
corr = data.corr()


print(corr)


# correlation threshold of 0.9.

# Based on the  correlation matrix, there are no pairs of variables in the dataset that are highly correlated. The highest correlation, between age and bmi, is only moderate at 0.358897.

# ## Box Plot

# In[27]:



plt.figure(figsize=(20, 15))  

numerical_cols = ['id', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke']

for i, column in enumerate(numerical_cols, 1):
    plt.subplot(3, 3, i) 
    sns.boxplot(y=data[column])
    plt.title(f'Box plot of {column}')

plt.tight_layout()  
plt.show()


# Imbalance: Variables like hypertension, heart_disease, and stroke are highly imbalanced, with the majority of values being 0.
# 
# Outliers: Significant outliers are present in avg_glucose_level and bmi.
# 
# Skewness: Both avg_glucose_level and bmi distributions are right-skewed.

# In[28]:


print(data.isnull().sum())


# In[30]:


data.dropna(subset=['bmi', 'smoking_status'], inplace=True)


# In[31]:


print(data.isnull().sum())


# # Removing duplicate rows

# In[33]:


print(f'Number of duplicate rows: {data.duplicated().sum()}')


# In[34]:


print("Data shape after removing missing values:", data.shape)


# # checking data types

# In[35]:


print(data.dtypes)


# In[36]:


from scipy import stats


# # DEALING WITH OUTLIERS 

# In[39]:


def cap_outliers(series):
    upper_limit = series.quantile(0.99)
    return np.where(series > upper_limit, upper_limit, series)

data['avg_glucose_level'] = cap_outliers(data['avg_glucose_level'])
data['bmi'] = cap_outliers(data['bmi'])


# # DEALING WITH SKEWNESS

# In[43]:


data['avg_glucose_level'] = np.log1p(data['avg_glucose_level'])
data['bmi'] = np.log1p(data['bmi'])


# In[44]:


print("\nClass Distribution:")
print(data['stroke'].value_counts())


# In[49]:


# Save to new CSV file
data.to_csv('C:\\Users\\Aman ur Rehman\\Documents\\stroke_prediction.csv', index=False)


# # ONE HOT ENCODING

# In[55]:


data_path = r'C:\\Users\\Aman ur Rehman\\Documents\\stroke_prediction.csv'
cleaned_data = pd.read_csv(data_path)

categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

# Perform one-hot encoding on categorical variables
cleaned_data_encoded = pd.get_dummies(cleaned_data, columns=categorical_columns)

# Separate features (X) and target variable (y)
X = cleaned_data_encoded.drop(columns=['stroke'])
y = cleaned_data_encoded['stroke']


# # PCA with 6 components

# In[54]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Standardize the features before applying PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

min_num_components = 6 
pca = PCA(n_components=min_num_components, svd_solver='full')  

# Fit PCA on the scaled features
X_pca = pca.fit_transform(X_scaled)

# Retrieve the names of the original features
feature_names = X.columns

components = pca.components_

# Create a DataFrame for components to see the contribution of each original feature
components_df = pd.DataFrame(components, columns=feature_names)

# Save PCA-transformed features to a new CSV file
pca_df = pd.DataFrame(data=X_pca, columns=[f'PC{i}' for i in range(1, min_num_components + 1)])
pca_df['stroke'] = y  # Add the target variable back
output_path = r'C:\\Users\\Aman ur Rehman\\Documents\\stroke_prediction_pca.csv'
pca_df.to_csv(output_path, index=False)

# Print summary
print(f"Number of features before PCA transformation: {X.shape[1]}")
print(f"Number of features after PCA transformation: {X_pca.shape[1]}")
print(f"Number of features removed by PCA: {X.shape[1] - X_pca.shape[1]}")

# Print explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
print(f"Explained variance ratio of the principal components: {explained_variance_ratio}")
print(f"Total explained variance by {min_num_components} components: {explained_variance_ratio.sum()}")

# Plot explained variance ratio
plt.figure(figsize=(8, 6))
plt.bar(range(1, min_num_components + 1), explained_variance_ratio, alpha=0.5, align='center')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio of Principal Components')
plt.show()


# # PCA with  components explaining 95% variance

# In[61]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

# Standardize the features before applying PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# variance ratio
desired_variance_ratio = 0.95
pca = PCA(n_components=desired_variance_ratio)

# Fit PCA on the scaled features
X_pca = pca.fit_transform(X_scaled)

# Retrieve the names of the original features
feature_names = X.columns

# Calculate the contribution of each original feature to the principal components
components = pca.components_

# Create a DataFrame for components to see the contribution of each original feature
components_df = pd.DataFrame(components, columns=feature_names)

# Save PCA-transformed features to a new CSV file
pca_df = pd.DataFrame(data=X_pca, columns=[f'PC{i}' for i in range(1, pca.n_components_ + 1)])
pca_df['stroke'] = y  
output_path = r'C:\\Users\\Aman ur Rehman\\Documents\\stroke_prediction_pca_1.csv'
pca_df.to_csv(output_path, index=False)

# Print summary
print(f"Number of features before PCA transformation: {X.shape[1]}")
print(f"Number of features after PCA transformation: {X_pca.shape[1]}")
print(f"Number of features removed by PCA: {X.shape[1] - X_pca.shape[1]}")

# Print explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
print(f"Explained variance ratio of the principal components: {explained_variance_ratio}")
print(f"Total explained variance by {X_pca.shape[1]} components: {explained_variance_ratio.sum()}")


# # Logistic Regression for feature selection

# In[59]:


from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
import pandas as pd

pca_data = pd.read_csv('C:\\Users\\Aman ur Rehman\\Documents\\stroke_prediction_pca_1.csv')

# Separate features (X) and target variable (y)
X = pca_data.drop(columns=['stroke'])  
y = pca_data['stroke']

#  Logistic Regression model with L1 penalty
logistic_reg = LogisticRegression(penalty='l1', solver='liblinear', C=1, random_state=42)

# Perform feature selection
selector = SelectFromModel(logistic_reg)
selector.fit(X, y)

selected_features = X.columns[selector.get_support()]

# Print selected features
print("Selected Features:")
print(selected_features)

# Transform the dataset to include only selected features
X_selected = selector.transform(X)

# Create a DataFrame with selected features
selected_data = pd.DataFrame(data=X_selected, columns=selected_features)
selected_data['Class'] = y  # Add the target variable back

# Save selected features to a new CSV file
selected_data.to_csv('C:\\Users\\Aman ur Rehman\\Documents\\selected_features_pca1.csv', index=False)


# L1 did not remove any features.This can be because all features in the PCA-transformed data might contribute significantly to the model, making it difficult for L1 regularization to zero out any coefficients.

# # Mutual information applied on 15 features to select top 10.

# In[65]:


from sklearn.feature_selection import mutual_info_classif

data = pd.read_csv('C:\\Users\\Aman ur Rehman\\Documents\\stroke_prediction_pca_1.csv')

# Separate features and target variable
X = data.drop(columns=['stroke'])
y = data['stroke']

# Compute mutual information scores
mi_scores = mutual_info_classif(X, y)

# Create a DataFrame to display feature names and their mutual information scores
mi_scores_df = pd.DataFrame({'Feature': X.columns, 'Mutual_Information': mi_scores})
mi_scores_df = mi_scores_df.sort_values(by='Mutual_Information', ascending=False)

# Display the DataFrame
print(mi_scores_df)

# Select the top N features
N = 10  
top_features = mi_scores_df.head(N)['Feature']

X_selected = X[top_features]

# Create a DataFrame with the selected features
selected_data = X_selected.copy()
selected_data['stroke'] = y  

# Save the selected features to a new CSV file
selected_data.to_csv('C:\\Users\\Aman ur Rehman\\Documents\\selected_features_mi1.csv', index=False)

# Display the selected features
print(f"Top {N} Features based on Mutual Information:")
print(top_features)


# In[ ]:





# # Using PCA to determine the variance explained by the top 10 features

# In[66]:


from sklearn.decomposition import PCA

selected_data = pd.read_csv('C:\\Users\\Aman ur Rehman\\Documents\\selected_features_mi1.csv')

# Separate features and target variable
X_selected = selected_data.drop(columns=['stroke'])
y = selected_data['stroke']

# Apply PCA to the selected features
pca = PCA()
X_pca = pca.fit_transform(X_selected)

# Calculate the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
total_explained_variance = explained_variance_ratio.cumsum()

print("Explained variance ratio by each principal component:")
print(explained_variance_ratio)

print("Cumulative explained variance by principal components:")
print(total_explained_variance)


# # MODEL TRAINING - RANDOM FOREST CLASSIFIER

# In[4]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the PCA-transformed dataset after top k selection
pca_data = pd.read_csv('C:\\Users\\Aman ur Rehman\\Documents\\selected_features_mi1.csv')

X_pca = pca_data.drop(columns=['stroke'])
y = pca_data['stroke']
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of Random Forest Classifier: {accuracy}")


# In[5]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Calculate precision
precision = precision_score(y_test, y_pred)
print(f"Precision: {precision}")

# Calculate recall
recall = recall_score(y_test, y_pred)
print(f"Recall: {recall}")

# Calculate F1 score
f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1}")

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# # CROSS VALIDATION

# In[6]:


from sklearn.model_selection import cross_val_score

# Initialize Random Forest classifier
rf_classifier = RandomForestClassifier()

# Perform cross-validation
cv_scores = cross_val_score(rf_classifier, X_train, y_train, cv=5)

# Print cross-validation scores
print("Cross-validation Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())


# # Logistic Regression with Cross-Validation

# In[7]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Initialize Logistic Regression classifier
lr_classifier = LogisticRegression()

# Perform cross-validation
y_pred = cross_val_predict(lr_classifier, X_train, y_train, cv=5)

# Calculate evaluation metrics
accuracy = accuracy_score(y_train, y_pred)
precision = precision_score(y_train, y_pred)
recall = recall_score(y_train, y_pred)
f1 = f1_score(y_train, y_pred)
roc_auc = roc_auc_score(y_train, y_pred)
conf_matrix = confusion_matrix(y_train, y_pred)

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("Confusion Matrix:\n", conf_matrix)


# # Support Vector Machine (SVM) with Cross-Validation

# In[8]:


from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Initialize SVM classifier
svm_classifier = SVC()

# Perform cross-validation
y_pred = cross_val_predict(svm_classifier, X_train, y_train, cv=5)

# Calculate evaluation metrics
accuracy = accuracy_score(y_train, y_pred)
precision = precision_score(y_train, y_pred)
recall = recall_score(y_train, y_pred)
f1 = f1_score(y_train, y_pred)
roc_auc = roc_auc_score(y_train, y_pred)
conf_matrix = confusion_matrix(y_train, y_pred)

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("Confusion Matrix:\n", conf_matrix)


# # Gradient Boosting Machine (XGBoost) with Cross-Validation

# In[9]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Initialize Gradient Boosting classifier
gb_classifier = GradientBoostingClassifier()

# Perform cross-validation
y_pred = cross_val_predict(gb_classifier, X_train, y_train, cv=5)

# Calculate evaluation metrics
accuracy = accuracy_score(y_train, y_pred)
precision = precision_score(y_train, y_pred)
recall = recall_score(y_train, y_pred)
f1 = f1_score(y_train, y_pred)
roc_auc = roc_auc_score(y_train, y_pred)
conf_matrix = confusion_matrix(y_train, y_pred)

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("Confusion Matrix:\n", conf_matrix)


# 
# # K-Nearest Neighbors (KNN) with Cross-Validation

# In[10]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Initialize KNN classifier
knn_classifier = KNeighborsClassifier()

# Perform cross-validation
y_pred = cross_val_predict(knn_classifier, X_train, y_train, cv=5)

# Calculate evaluation metrics
accuracy = accuracy_score(y_train, y_pred)
precision = precision_score(y_train, y_pred)
recall = recall_score(y_train, y_pred)
f1 = f1_score(y_train, y_pred)
roc_auc = roc_auc_score(y_train, y_pred)
conf_matrix = confusion_matrix(y_train, y_pred)

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("Confusion Matrix:\n", conf_matrix)


# In[ ]:





# # We will apply SMOTE, One-Class Learning, and Cluster-Based Oversampling—to improve model performance in the face of class imbalance. These will be paired with three machine learning algorithms: Logistic Regression, SVM, and XGBoost.

# # SMOTE

# # Application with Logistic Regression, SVM, and XGBoost:
# 

# In[12]:


get_ipython().system('pip install -U scikit-learn imbalanced-learn')


# In[13]:


try:
    from imblearn.over_sampling import SMOTE
    print("SMOTE is ready to be used.")
except ImportError as e:
    print("Failed to import SMOTE:", e)


# # Logistic Regression

# In[15]:


get_ipython().system('pip install imbalanced-learn')


from sklearn.metrics import classification_report

from imblearn.over_sampling import SMOTE

# Load the PCA-transformed dataset
pca_data = pd.read_csv('C:\\Users\\Aman ur Rehman\\Documents\\selected_features_mi1.csv')

X = pca_data.drop(columns=['stroke'])  
y = pca_data['stroke']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Initialize logistic regression classifier
logreg_classifier = LogisticRegression()

# Fit the logistic regression model on SMOTE-transformed data
logreg_classifier.fit(X_train_smote, y_train_smote)

# Predict on the test set
y_pred = logreg_classifier.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("Confusion Matrix:\n", conf_matrix)


# # SVM

# In[17]:


from sklearn.svm import SVC



# Initialize SVM classifier
svm_classifier = SVC()

# Fit the SVM model
svm_classifier.fit(X_train_smote, y_train_smote)

# Predict on the test set
y_pred = svm_classifier.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print evaluation metrics
print("SVM Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("Confusion Matrix:\n", conf_matrix)


# # XGBOOST

# In[19]:


from xgboost import XGBClassifier

# Initialize XGBoost classifier
xgb_classifier = XGBClassifier()

# Fit the XGBoost model
xgb_classifier.fit(X_train_smote, y_train_smote)

# Predict on the test set
y_pred = xgb_classifier.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print evaluation metrics
print("XGBoost Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("Confusion Matrix:\n", conf_matrix)


# # One-Class Learning

# In[20]:



from sklearn.ensemble import IsolationForest



# Initialize Isolation Forest classifier
isolation_forest = IsolationForest(contamination=0.1, random_state=42)

# Fit the Isolation Forest model
isolation_forest.fit(X_train)

# Predict on the test set
y_pred = isolation_forest.predict(X_test)

# Convert predictions to binary labels (1 for inliers, -1 for outliers)
y_pred_binary = [1 if x == 1 else 0 for x in y_pred]

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)
roc_auc = roc_auc_score(y_test, y_pred_binary)
conf_matrix = confusion_matrix(y_test, y_pred_binary)

# Print evaluation metrics
print("Isolation Forest Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("Confusion Matrix:\n", conf_matrix)


# # Logistic Regression with Class Weighting

# In[21]:


# Class weights
class_weights = {0: 1, 1: 10} 

# Logistic Regression with class weighting
logreg_classifier = LogisticRegression(class_weight=class_weights)
logreg_classifier.fit(X_train, y_train)
y_pred_logreg = logreg_classifier.predict(X_test)

# Evaluation metrics for Logistic Regression
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
precision_logreg = precision_score(y_test, y_pred_logreg)
recall_logreg = recall_score(y_test, y_pred_logreg)
f1_logreg = f1_score(y_test, y_pred_logreg)
roc_auc_logreg = roc_auc_score(y_test, y_pred_logreg)
conf_matrix_logreg = confusion_matrix(y_test, y_pred_logreg)

# Print evaluation metrics for Logistic Regression
print("Logistic Regression Metrics:")
print("Accuracy:", accuracy_logreg)
print("Precision:", precision_logreg)
print("Recall:", recall_logreg)
print("F1 Score:", f1_logreg)
print("ROC AUC Score:", roc_auc_logreg)
print("Confusion Matrix:\n", conf_matrix_logreg)


# # Support Vector Machine (SVM) with Class Weighting

# In[22]:



# SVM with class weighting
svm_classifier = SVC(class_weight=class_weights)
svm_classifier.fit(X_train, y_train)
y_pred_svm = svm_classifier.predict(X_test)

# Evaluation metrics for SVM
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)
roc_auc_svm = roc_auc_score(y_test, y_pred_svm)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)

# Print evaluation metrics for SVM
print("SVM Metrics:")
print("Accuracy:", accuracy_svm)
print("Precision:", precision_svm)
print("Recall:", recall_svm)
print("F1 Score:", f1_svm)
print("ROC AUC Score:", roc_auc_svm)
print("Confusion Matrix:\n", conf_matrix_svm)


# # Random Forest with Class Weighting

# In[23]:


rf_classifier = RandomForestClassifier(class_weight=class_weights, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)

# Evaluation metrics for Random Forest
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

# Print evaluation metrics for Random Forest
print("Random Forest Metrics:")
print("Accuracy:", accuracy_rf)
print("Precision:", precision_rf)
print("Recall:", recall_rf)
print("F1 Score:", f1_rf)
print("ROC AUC Score:", roc_auc_rf)
print("Confusion Matrix:\n", conf_matrix_rf)


# ## Pipeline for comparing baseline model after feature selection.

# In[2]:



from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate

data = pd.read_csv('C:\\Users\\Aman ur Rehman\\Documents\\selected_features_mi1.csv')

X = data.drop(columns=['stroke'])
y = data['stroke']

classifiers = {
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Support Vector Machine': SVC(probability=True),
    'XGBoost': XGBClassifier(eval_metric='logloss'),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1': make_scorer(f1_score, average='weighted'),
    'roc_auc': make_scorer(roc_auc_score, average='weighted')
}

pipelines = {}
for name, clf in classifiers.items():
    if name in ['Support Vector Machine', 'Logistic Regression', 'K-Nearest Neighbors']:
        # SVM, Logistic Regression, and KNN require feature scaling
        pipelines[name] = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', clf)
        ])
    else:
        pipelines[name] = Pipeline([
            ('classifier', clf)
        ])

# cross-validation
results = {}
for name, pipeline in pipelines.items():
    cv_results = cross_validate(pipeline, X, y, cv=5, scoring=scoring, return_train_score=False)
    results[name] = cv_results

# Print average scores for each classifier
for name, scores in results.items():
    print(f"\nClassifier: {name}")
    for metric in scoring:
        print(f"{metric}: {np.mean(scores['test_' + metric]):.4f} (std: {np.std(scores['test_' + metric]):.4f})")


# In[3]:




# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fit and plot ROC AUC curves for each classifier
plt.figure(figsize=(10, 8))

for name, pipeline in pipelines.items():
    # Fit the model on the training data
    pipeline.fit(X_train, y_train)
    
    # Predict probabilities on the test data
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Compute ROC curve and ROC AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot the ROC curve
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

# Plot settings
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC AUC Curves')
plt.legend(loc='lower right')
plt.show()


# The ROC AUC values being close to 0.5 suggest that the classifiers might not be effectively distinguishing between the positive and negative classes. This could be due to class imbalance or other issues in the data.
# The very high and similar accuracy, precision, recall, and F1-scores indicate that the classifiers are performing similarly well on the dataset.

# ## Pipeline for comparing  model after feature selection and class imbalance algorithms.

# In[1]:


import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# In[2]:


from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# In[3]:


data = pd.read_csv('C:\\Users\\Aman ur Rehman\\Documents\\selected_features_mi1.csv')

X = data.drop(columns=['stroke'])
y = data['stroke']


# In[4]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[6]:


classifiers = {
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Support Vector Machine': SVC(probability=True),
    'XGBoost': XGBClassifier(eval_metric='logloss'),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1': make_scorer(f1_score, average='weighted'),
    'roc_auc': make_scorer(roc_auc_score, average='weighted')
}


# ## Smote pipeline

# In[7]:


pipelines_smote = {}
for name, clf in classifiers.items():
    if name in ['Support Vector Machine', 'Logistic Regression', 'K-Nearest Neighbors']:
        pipelines_smote[name + ' (SMOTE)'] = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('classifier', clf)
        ])
    else:
        pipelines_smote[name + ' (SMOTE)'] = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('classifier', clf)
        ])


# ## Class weight pipeline

# In[8]:


class_weights = {0: 1, 1: 10}  
pipelines_class_weight = {
    'Random Forest (Class Weight)': RandomForestClassifier(class_weight=class_weights, random_state=42),
    'Logistic Regression (Class Weight)': LogisticRegression(class_weight=class_weights, max_iter=1000),
    'Support Vector Machine (Class Weight)': SVC(class_weight=class_weights, probability=True)
}


# In[9]:


all_pipelines = {**pipelines_smote, **pipelines_class_weight}


# In[10]:


results = {}
for name, pipeline in all_pipelines.items():
    cv_results = cross_validate(pipeline, X, y, cv=5, scoring=scoring, return_train_score=False)
    results[name] = cv_results

# Print average scores for each classifier
for name, scores in results.items():
    print(f"\nClassifier: {name}")
    for metric in scoring:
        print(f"{metric}: {np.mean(scores['test_' + metric]):.4f} (std: {np.std(scores['test_' + metric]):.4f})")


# The stroke prediction dataset comprises features related to individuals' health parameters and lifestyle factors, aiming to predict the likelihood of stroke occurrence.
# Baseline Models Comparison
# 
# The baseline classifiers considered for this analysis include Random Forest, Logistic Regression, Support Vector Machine (SVM), XGBoost, and K-Nearest Neighbors (KNN).
# 
# From the baseline results, we observe that all classifiers achieved high accuracy, precision, recall, and F1 scores. However, there is a slight variation in performance metrics across different classifiers, with Logistic Regression and SVM showing marginally higher performance compared to others.

# 
# The performance of CI solutions significantly varies by the choice of different algorithms. The choice of algorithm impacts how well the model handles class imbalance, as seen in the differing metrics:
# •	Random Forest: Performs well with and without CI solutions, with SMOTE improving the balance between precision and recall.
# •	Logistic Regression: Performance significantly drops with SMOTE, but class weighting improves balance without major performance loss.
# •	Support Vector Machine: Similar trends to Logistic Regression, with SMOTE improving balance at a cost, while class weighting maintains good performance.
# •	XGBoost: Shows good balance improvements with SMOTE but at the cost of accuracy.
# •	K-Nearest Neighbors: Shows significant improvements in balance with SMOTE, though not as efficient as tree-based methods.
# In conclusion, while CI techniques improve balance between precision and recall, their effectiveness varies with the choice of algorithm. Tree-based methods like Random Forest and XGBoost generally perform better in handling class imbalance compared to linear methods like Logistic Regression and SVM.
# 

# In[ ]:




