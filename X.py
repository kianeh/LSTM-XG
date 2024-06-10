from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from keras.utils import to_categorical
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, Normalizer, MaxAbsScaler
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from imblearn.over_sampling import SMOTE
from keras.utils import to_categorical
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, Normalizer, MaxAbsScaler
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import pickle
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from keras.utils import to_categorical
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, Normalizer, MaxAbsScaler
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score ,log_loss
from keras.models import Sequential
# Existing imports...

class clfr:
    def __init__(self):
        self.pipeline = self.init_pipeline()
        self.x_train, self.x_test, self.y_train, self.y_test = self.data_reader()

    # Existing methods...

    def data_reader(self):
        df = pd.read_csv(r"C:\Users\kiane\BankChurners.csv")
        df = df[['CLIENTNUM', 'Attrition_Flag', 'Customer_Age', 'Gender',
                 'Dependent_count', 'Education_Level', 'Marital_Status',
                 'Income_Category', 'Card_Category', 'Months_on_book',
                 'Total_Relationship_Count', 'Months_Inactive_12_mon',
                 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']]

        # Categorical columns
        cat_cols = ['Gender', 'Marital_Status', 'Education_Level', 'Dependent_count',
                    'Income_Category', 'Card_Category', 'Total_Relationship_Count',
                    'Months_Inactive_12_mon', 'Contacts_Count_12_mon']

        # Map Attrition_Flag to integers
        af_unq = df['Attrition_Flag'].unique()
        af_map = {af_unq[i]: i for i in range(len(af_unq))}
        df['Attrition_Flag'] = df['Attrition_Flag'].map(af_map)

        # Separate features and target
        X = df.drop(["CLIENTNUM", "Attrition_Flag"], axis=1)
        y = df["Attrition_Flag"]

        # Train-test split
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=2)

        return x_train, x_test, y_train, y_test

    # Existing methods...

    def init_pipeline(self):
        num_transformer = Pipeline(steps=[('scaler', StandardScaler())]) # Scale Numerical values
        cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))]) # One-hot encode categorical values

        # Using ColumnTransformers to Select different column Types
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, selector(dtype_include=['float64', 'int64'])),
                ('cat', cat_transformer, selector(dtype_include=['object']))])

        my_pipeline = imbpipeline(steps=[
                                        ('preprocessor', preprocessor),
                                        ('smote', SMOTE(random_state=2)), # Because the Dataset is unbalanced Use SMOTE for Oversampling and making the Train Dataset balanced
                                        ('classifier', XGBClassifier())
                                        ])
        
        return my_pipeline    
    # Existing code...

    def evaluate_classifier(self, classifier_name, GridSearch_parameters):
        print(f"------------\n   {classifier_name}   \n------------")

        # Define the parameter grid for GridSearchCV
        param_grid = {
            'classifier': [eval(classifier_name)()],
            'preprocessor__num__scaler': [StandardScaler()],
            **GridSearch_parameters
        }

        # Create StratifiedKFold
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)

        # Create lists to store metrics
        train_accuracy_list = []
        test_accuracy_list = []
        val_accuracy_list = []
        val_loss_list = []

        for train_index, test_index in stratified_kfold.split(self.x_train, self.y_train):
            # Split data into train and validation sets
            X_train_fold, X_val_fold = self.x_train.iloc[train_index], self.x_train.iloc[test_index]
            y_train_fold, y_val_fold = self.y_train.iloc[train_index], self.y_train.iloc[test_index]

            # Fit the pipeline to the training fold
            grid_search = GridSearchCV(
                estimator=self.pipeline,
                param_grid=param_grid,
                cv=stratified_kfold,
                refit=True,
                n_jobs=-1,
            )
            grid_search.fit(X_train_fold, y_train_fold)

            # Evaluate on training fold
            y_preds_train_fold = grid_search.predict(X_train_fold)
            train_accuracy_fold = accuracy_score(y_train_fold, y_preds_train_fold)

            # Evaluate on test fold
            y_preds_test_fold = grid_search.predict(self.x_test)
            test_accuracy_fold = accuracy_score(self.y_test, y_preds_test_fold)

            # Evaluate on validation fold
            y_preds_val_fold = grid_search.predict(X_val_fold)
            val_accuracy_fold = accuracy_score(y_val_fold, y_preds_val_fold)

            # Calculate validation loss
            y_preds_proba_val_fold = grid_search.predict_proba(X_val_fold)
            val_loss_fold = log_loss(y_val_fold, y_preds_proba_val_fold)

            # Append metrics to lists
            train_accuracy_list.append(train_accuracy_fold)
            test_accuracy_list.append(test_accuracy_fold)
            val_accuracy_list.append(val_accuracy_fold)
            val_loss_list.append(val_loss_fold)

        # Calculate average metrics
        avg_train_accuracy = sum(train_accuracy_list) / len(train_accuracy_list)
        avg_test_accuracy = sum(test_accuracy_list) / len(test_accuracy_list)
        avg_val_accuracy = sum(val_accuracy_list) / len(val_accuracy_list)
        avg_val_loss = sum(val_loss_list) / len(val_loss_list)

        # Print and return metrics
        print("Training Accuracy:", avg_train_accuracy)
        print("Test Accuracy:", avg_test_accuracy)
        print("Validation Accuracy:", avg_val_accuracy)
        print("Validation Loss:", avg_val_loss)
        # Plot the validation loss
        plt.plot(val_loss_list, label='Validation Loss')
        plt.xlabel('Fold')
        plt.ylabel('Loss')
        plt.title('Validation Loss for each Fold')
        plt.legend()
        plt.show()
         # Plot the validation accuracy
        plt.plot(val_accuracy_list, label='Validation Accuracy')
        plt.xlabel('Fold')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy for each Fold')
        plt.legend()
        plt.show()
       
        return avg_train_accuracy, avg_test_accuracy

# Create an instance of the class
_cls = clfr()

# Example usage for XGBoost classifier
xgboost_params = {
    'classifier__C': [5, 10],
    'classifier__degree': [3],
    'classifier__gamma': [0.1],
    'classifier__kernel': ['rbf'],
}

acc_train, acc_test = _cls.evaluate_classifier('XGBClassifier', xgboost_params)
print("Training Accuracy:", acc_train)
print("Test Accuracy:", acc_test)

