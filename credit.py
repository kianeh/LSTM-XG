# loading libaries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer   
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve,auc
from sklearn.metrics import make_scorer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as imbpipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import lightgbm as ltb
import pyforest
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder, StandardScaler, PowerTransformer, MinMaxScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import RepeatedStratifiedKFold, KFold, cross_val_predict, train_test_split, GridSearchCV, cross_val_score, cross_validate
from sklearn.linear_model import LinearRegression, Lasso, Ridge,ElasticNet
from sklearn.metrics import plot_confusion_matrix, r2_score, mean_absolute_error, mean_squared_error, classification_report, confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import make_scorer, precision_score, precision_recall_curve, plot_precision_recall_curve, plot_roc_curve, roc_auc_score, roc_curve, f1_score, accuracy_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostClassifier
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_classif, f_regression, mutual_info_regression
from xgboost import XGBRegressor, XGBClassifier
from xgboost import plot_importance
from sklearn.pipeline import Pipeline
from sklearn.tree import plot_tree
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import cufflinks as cf
import plotly.offline
from scipy.stats import skew
from coalas import csvReader as c
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
def convert(data):
    number = preprocessing.LabelEncoder()
import warnings
warnings.filterwarnings('ignore')
warnings.warn("this will not show")

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#loading dataset
data=pd.read_csv(r"C:\Users\kiane\BankChurners.csv")  
print(data)
data.columns
print(data.columns)
data=data[['CLIENTNUM', 'Attrition_Flag', 'Customer_Age', 'Gender',
          'Dependent_count', 'Education_Level', 'Marital_Status',
          'Income_Category', 'Card_Category', 'Months_on_book',
          'Total_Relationship_Count', 'Months_Inactive_12_mon',
          'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
          'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
          'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']]
data.head()    
data.info()
df=pd.DataFrame(data)
print(df.describe())
df.describe(include=['O'])
print(df.describe(include=['O']))

#data missing
data.isnull().sum()
print(df.isnull())

#data duplication
s = df.duplicated()
print(df.duplicated())

#exploratory data analysis(data distrbution & data graph)
plt.style.use('ggplot')
plt.rcParams['figure.figsize']=(10,8)
inter=data['Attrition_Flag'].value_counts()
plt.pie(inter,labels=inter.index,autopct='%0.1f%%',explode=[0.0,0.2]);
plt.show()
def plot_num(var):
    
    plt.subplot(1,2,1)
    sns.histplot(data=data,x=var,hue='Attrition_Flag',kde=True);
    
    plt.subplot(1,2,2)
    sns.boxplot(x='Attrition_Flag',y=var,data=data);
plot_num('Customer_Age')
plt.show()
plot_num('Credit_Limit')
plt.show()
plot_num('Avg_Utilization_Ratio')
plt.show()
plot_num('Months_on_book')
plt.show()
plot_num('Total_Trans_Amt')
plt.show()
plot_num('Total_Revolving_Bal')
plt.show()
plot_num('Avg_Open_To_Buy')
plt.show()
plot_num('Total_Trans_Ct')
plt.show()
plot_num('Total_Amt_Chng_Q4_Q1')
plt.show()
plot_num('Total_Ct_Chng_Q4_Q1')
plt.show()

#Total_Trans_Ct, Total_Trans_Amt, Total_Revolving_Bal,Total_Ct_Chng_Q4_Q1 and Avg_Utilization_Ratio seems effective on decision making
#Age and month_on_book variable are irrevalant
num=['Avg_Utilization_Ratio','Customer_Age','Credit_Limit','Months_on_book',
     'Total_Revolving_Bal','Total_Trans_Amt','Avg_Open_To_Buy','Total_Trans_Ct',
     'Total_Amt_Chng_Q4_Q1', 'Total_Ct_Chng_Q4_Q1']
df = pd.DataFrame(np.random.rand(10,10))    
sns.heatmap(data[num].corr(),cbar=False,annot=True)
plt.show()
col=['Avg_Utilization_Ratio','Total_Revolving_Bal','Total_Trans_Amt',
     'Avg_Open_To_Buy','Total_Trans_Ct','Total_Amt_Chng_Q4_Q1', 'Total_Ct_Chng_Q4_Q1',
     'Attrition_Flag']

#attrition flag shows account activity(1=inactive 0=active)   
sns.pairplot(data[col],hue='Attrition_Flag');
plt.show()
sns.lmplot(x='Total_Trans_Amt',y='Total_Trans_Ct',hue='Attrition_Flag',data=data);
plt.show()
sns.lmplot(x='Avg_Utilization_Ratio',y='Avg_Open_To_Buy',hue='Attrition_Flag',data=data);
plt.show()

#There are non-linear relationship between target variable and input variable
cat=['Gender','Marital_Status','Education_Level','Dependent_count','Attrition_Flag',
     'Income_Category', 'Card_Category', 'Total_Relationship_Count', 'Months_Inactive_12_mon',
     'Contacts_Count_12_mon' ]
def call():
    
    for i in range(len(cat)):
        
        print(data[cat[i]].unique())
        print(data[cat[i]].value_counts())
        print("**"*25+'\n')
        
call()    
data['Attrition_Flag']=data['Attrition_Flag'].replace({'Existing Customer':0,'Attrited Customer':1})
def plot_cat(var):
    
    inter=data.groupby(var)['Attrition_Flag'].mean()
    order=inter.index
    
    plt.subplot(1,2,1)
    sns.countplot(x=var,data=data,order=order);
    
    
    plt.subplot(1,2,2)
    plt.pie(inter,labels=inter.index,autopct="%0.1f%%",radius=1.5);
    
    
    plt.tight_layout()

plot_cat('Gender')
plt.show()  
plot_cat('Total_Relationship_Count')
plt.show()
plot_cat('Contacts_Count_12_mon')
plt.show()
plot_cat('Months_Inactive_12_mon')
plt.show()

#data preprocessing
#Remove rows or columns by specifying label
X=data.drop(['CLIENTNUM','Attrition_Flag'],axis=1)
y=data['Attrition_Flag']

#select columns by their data types
cat=X.select_dtypes(include='object').columns
print(cat)

#Convert categorical variable into dummy/indicator variables
#set drop_first = True , then it will drop the first category
X_cat=pd.get_dummies(X[cat],drop_first=True)
num=X.select_dtypes(exclude='object').columns
print(num)

#increase the symmetry of the distribution of the features.
transformer=PowerTransformer()
X_num=transformer.fit_transform(X[num])
X_num=pd.DataFrame(X_num,columns=num)
X=pd.concat([X_cat,X_num],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y,
                                                    random_state=42)

#represents the probability that a random positive or random negative . AUC ranges in value from 0 to 1
def auc_score(y_true,y_probs):
    precision,recall,_=precision_recall_curve(y_true,y_probs)
    return auc(recall,precision)
metric=make_scorer(auc_score,needs_proba=True)

#selection importance feature method that uses estimator to determine the importance of the variable in determining the value of target variable.Because the data contains non-linear,multi-variate complex relations we will use random forest and xgboost
fs_rf=SelectFromModel(RandomForestClassifier()).fit(X_train,y_train)
imp_rf=pd.Series(fs_rf.estimator_.feature_importances_,index=X_train.columns).sort_values(ascending=False)
imp_rf.plot(kind='barh')
plt.show()
fs_xgb=SelectFromModel(XGBClassifier()).fit(X_train,y_train)
imp_xgb=pd.Series(fs_xgb.estimator_.feature_importances_,index=X_train.columns).sort_values(ascending=False)
imp_xgb.plot(kind='barh')
plt.show()
features= ['Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1',
           'Total_Revolving_Bal', 'Avg_Utilization_Ratio', 'Total_Amt_Chng_Q4_Q1',
           'Total_Relationship_Count', 'Avg_Open_To_Buy', 'Credit_Limit',
           'Customer_Age', 'Contacts_Count_12_mon', 'Months_on_book',
           'Months_Inactive_12_mon', 'Dependent_count'] 
X_train=X_train[features]
X_test=X_test[features]

#modelling for imbalanced classification data
def get_models():
    
    models=[]
    names=[]
    
    models.append(LogisticRegression())
    names.append('lr')               
                  
    models.append(KNeighborsClassifier())
    names.append('knn')
    
    models.append(SVC(probability=True))
    names.append('svc')
    
    models.append(RandomForestClassifier())
    names.append('rf')
    
    models.append(AdaBoostClassifier())
    names.append('adb')
    
    models.append(XGBClassifier())
    names.append('xgb')

    return models,names
def evaluate(X,y):
    
    models,names=get_models()
    results=[]
    
    for i in range(len(models)):
        cv=StratifiedKFold(n_splits=5)
        scores=cross_val_score(models[i],X,y,scoring=metric,cv=cv)
        
        results.append(scores)
        
    plt.boxplot(results,labels=names,showmeans=True)
evaluate(X_train,y_train)

# evaluate a model
def evaluate_model(X, y, model):
    # define evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate model
    scores = cross_val_score(model, X, y, scoring='recall', cv=cv, n_jobs=-1)
    return scores
resample=SMOTEENN(enn=EditedNearestNeighbours(sampling_strategy='majority'))
X_rs,y_rs=resample.fit_resample(X_train,y_train)
evaluate(X_rs,y_rs)
# divide features and label , encode label
y = data['Attrition_Flag'].values
X = data.drop(columns = ['Attrition_Flag'])

#categorical and numerical 
num_ix = X.select_dtypes(include=['int64', 'float64']).columns
cat_ix = X.select_dtypes(include=['object', 'bool']).columns

# define models
models, names = get_models()
results = list()

#define pipeline for categorical transformer 
from sklearn.compose import ColumnTransformer
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))])

#split train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101, stratify=y)

#smote for oversampling
smt = SMOTE(random_state=42)
for i in range(len(models)):
    # define steps
    steps = [('c',categorical_transformer,cat_ix), ('n',MinMaxScaler(),num_ix)]
    # one hot encode categorical, normalize numerical
    ct = ColumnTransformer(steps)
    # wrap the model i a pipeline
    pipeline = imbpipeline(steps=[('t',ct),('smt', smt),('m',models[i])])
    # evaluate the model and store results
    scores = evaluate_model(X_train, y_train, pipeline)
    results.append(scores)
    # summarize performance
    from statistics import mean
print('>%s %.3f (%.3f)' % (names[i], np.mean(scores), np.std(scores)))  
# plot the results
plt.figure(figsize=(15, 10))
plt.boxplot(results, labels=names, showmeans=True)
plt.show()

#LightGBM is a gradient boosting framework.that uses tree based learning algorithms. It is designed to be distributed and efficient
param_grid = \
[
    {  
    "m__learning_rate":[0.1, 0.5],
    "m__max_depth":[30],
    "m__num_leaves":[10,100],
    "m__feature_fraction":[0.1,1.0],
    "m__subsample":[0.1,1.0],
    }
]
param_grid
#disable warning of model
import warnings
warnings.filterwarnings("ignore")
smt = SMOTE(random_state=42)

#BGC = BaggingClassifier()
CLS = LGBMClassifier()

# define steps
steps = [('c',categorical_transformer,cat_ix), ('n',MinMaxScaler(),num_ix)]
# one hot encode categorical, normalize numerical
ct = ColumnTransformer(steps)
# wrap the model i a pipeline
pipeline = imbpipeline(steps=[('t',ct),('smt',smt),('m',CLS)])
# evaluate the model and store results
grid_search = GridSearchCV(pipeline, param_grid, cv=5,
                       scoring='recall',
                       return_train_score=True,n_jobs = -1, refit=True)

grid_search.fit(X_train, y_train)
import numpy as np
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_score, params)
#best parameter
print(grid_search.best_score_)
grid_search.best_params_  
#test final model
name = "LGBMClassifier"
smt = SMOTE(random_state=42)
CLS = LGBMClassifier(feature_fraction = 1.0,learning_rate=0.5,
                         max_depth= 30,num_leaves= 10,subsample= 0.1)

# define steps
steps = [('c',categorical_transformer,cat_ix), ('n',MinMaxScaler(),num_ix)]
# one hot encode categorical, normalize numerical
ct = ColumnTransformer(steps)
# wrap the model i a pipeline
pipeline = imbpipeline(steps=[('t',ct),('smt',smt),('m',CLS)])
# evaluate the model and store results
pipeline.fit(X_train, y_train)
#predict
preds = pipeline.predict(X_test)
preds

#evaluate final model
# Print the prediction accuracy
pred_precision = metrics.precision_score (y_test, preds)
pred_accuracy = metrics.accuracy_score(y_test, preds)
pred_recall = metrics.recall_score (y_test, preds)
print('Precision: ' f"{pred_precision:,.3%}")
print('Accuracy: ' f"{pred_accuracy:,.3%}")
print('Recall: ' f"{pred_recall:,.3%}")
confusion_matrix(y_test,preds)

#confusion matrix
cfm = confusion_matrix(y_test, preds)
plt.figure(figsize=(10,10))
sns.heatmap(cfm, annot=True)
plt.xlabel('Predicted classes')
plt.ylabel('Actual classes')

#define important feature
feature_imp = pd.DataFrame(sorted(zip(CLS.feature_importances_,X.columns)), columns=['Value','Feature'])
plt.figure(figsize=(18, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.show()
