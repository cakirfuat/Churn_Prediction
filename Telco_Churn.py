########################################################
#Görev 1 : Keşifçi Veri Analizi
#################################################
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import Ready as read
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_ = pd.read_csv('.venv/lib/Data/Telco-Customer-Churn.csv')
df = df_.copy()

def what_the_data(dataframe):

    print("* İLK 10 GÖZLEM *")
    print("--------------------------------------------------------------------------")
    print(dataframe.head(10))

    print("--------------------------------------------------------------------------")
    print("* DEĞİŞKEN İSİMLERİ *")
    print("--------------------------------------------------------------------------")
    for i in dataframe.columns:
        print(i , "\n")
    print("--------------------------------------------------------------------------")
    print("* BETİMSEL İSTATİSTİK *")
    print("--------------------------------------------------------------------------")
    print(dataframe.describe().T)

    print("--------------------------------------------------------------------------")
    print("* KAYIP,BOŞ GÖZLEM *")
    print("--------------------------------------------------------------------------")
    print(dataframe.isnull().sum())

    print("--------------------------------------------------------------------------")
    print("* DEĞİŞKEN TİPLERİ *")
    print("--------------------------------------------------------------------------")
    print(dataframe.info())

    print("--------------------------------------------------------------------------")
    print("* VERİ BOYUTU *")
    print("--------------------------------------------------------------------------")
    print(dataframe.shape)
    print("Gözlem Birimi :" , dataframe.shape[0])
    print("Değişken Sayısı :", dataframe.shape[1])

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Info #####################")
    print(dataframe.info())
    print("##################### Desc #####################")
    print(dataframe.describe().T)

what_the_data(df)

check_df(df)

########################################################
# Adım 1: Numerikvekategorikdeğişkenleriyakalayınız.
#################################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}\n{cat_cols}')
    print(f'num_cols: {len(num_cols)}\n{num_cols}')
    print(f'cat_but_car: {len(cat_but_car)}\n{cat_but_car}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

########################################################
# Adım 2: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)
########################################################
df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
df['TotalCharges'] = df['TotalCharges'].astype(float).fillna(df['TotalCharges'].astype(float).mean())
df['TotalCharges'] = df['TotalCharges'].astype(float)
df['TotalCharges'].sort_values(ascending=False)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors="coerce")
df['SeniorCitizen'] = df['SeniorCitizen'].astype("O")
df['Churn'] = df['Churn'].replace({'No': 0, 'Yes': 1}).astype("O")

########################################################
# Adım 3:  Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.
########################################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    data = dataframe[numerical_col].describe(quantiles).T
    print(data)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df,col,plot=False)

for col in num_cols:
    num_summary(df,col,plot=False)

########################################################
# Adım 4: Kategorik değişkenler ile hedef değişken incelemesini yapınız.
########################################################

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].value_counts(normalize=True)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df,'Churn',col)

for col in num_cols:
    target_summary_with_num(df,'Churn',col)

########################################################
# Adım 5: Aykırı gözlem var mı inceleyiniz.
########################################################

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col, 0.05, 0.95))

########################################################
# Adım 6: Eksik gözlem var mı inceleyiniz.
########################################################

df.isnull().sum()

# Function to create a summary table of missing values in the dataframe.
def missing_values_table(dataframe, na_name=False):
    # Identify columns with missing values.
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    # Count missing values and calculate their percentage.
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    # Create a summary table.
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    # Optionally return the names of columns with missing values.
    if na_name:
        return na_columns


# Execute the function to display the missing values table.
na_columns = missing_values_table(df, na_name=True)

########################################################
#Görev 2 : Feature EngineeringAdım
# 1:  Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.
# ########################################################


# Adım# 2: Yeni değişkenler oluşturunuz.
check_df(df)


df['NEW_TOTALSERVICE'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                       'StreamingTV', 'StreamingMovies']] !="No").sum(axis=1)


cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Adım 3:  Encoding işlemlerini gerçekleştiriniz.Adım

cat_cols = [col for col in cat_cols if col not in ["Churn"]]

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    # Convert categorical variable into dummy/indicator variables.
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

# LABEL ENCODING: Transforming binary categorical variables into a machine-readable format.
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


# Identifying binary categorical columns for label encoding.
binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]

# Applying label encoding to binary columns.
for col in binary_cols:
    df = label_encoder(df, col)

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# 4: Numerik değişkenler için standartlaştırma yapınız.
rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])

#Görev 3 : Modelleme
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, roc_auc_score,classification_report, confusion_matrix
from sklearn.metrics import RocCurveDisplay
from sklearn.neighbors import KNeighborsClassifier
# Adım 1:  Sınıflandırma algoritmaları ile modeller kurup,
y = df["Churn"]
X = df.drop(["Churn", "customerID"], axis=1)  # Drop 'customerID' as it's not a feature.

models = [
    ('LR', LogisticRegression(random_state=12345)),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier(random_state=12345)),
    ('RF', RandomForestClassifier(random_state=12345)),
    ('SVM', SVC(gamma='auto', random_state=12345)),
    ('XGB', XGBClassifier(random_state=12345)),
    ("CatBoost", CatBoostClassifier(verbose=False, random_state=12345))
]
# accuracyskorlarını inceleyip. En iyi 4 modeli seçiniz.

for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")
# Adım 2: Seçtiğiniz modeller ile hiperparametreoptimizasyonu gerçekleştirin ve

# Initialize a Random Forest classifier.
rf_model = RandomForestClassifier(random_state=17)

# Define the parameter grid for Random Forests.
rf_params = {
    "max_depth": [5, 8, None],
    "max_features": [3, 5, 7, "auto"],
    "min_samples_split": [2, 5, 8, 15, 20],
    "n_estimators": [100, 200, 500]
}

# Perform grid search with cross-validation to find the best parameters.
rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

# Output the best parameters and the best score achieved.
print(rf_best_grid.best_params_)
print(rf_best_grid.best_score_)
# bulduğunuz hiparparametrelerile modeli tekrar kurunuz.


# Output the best parameters and the best score achieved.
print(rf_best_grid.best_params_)
print(rf_best_grid.best_score_)

# Train the final model with the best parameters.
rf_final: object = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)

# Evaluate the final model using cross-validation.
cv_results = cross_validate(rf_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
print(f"Random Forests Accuracy: {cv_results['test_accuracy'].mean()}")
print(f"Random Forests F1: {cv_results['test_f1'].mean()}")
print(f"Random Forests ROC AUC: {cv_results['test_roc_auc'].mean()}")

################################################
# XGBoost Hyperparameter Tuning
################################################

# Initialize an XGBoost classifier.
xgboost_model = XGBClassifier(random_state=17)

# Define the parameter grid for XGBoost.
xgboost_params = {
    "learning_rate": [0.1, 0.01, 0.001],
    "max_depth": [5, 8, 12, 15, 20],
    "n_estimators": [100, 500, 1000],
    "colsample_bytree": [0.5, 0.7, 1]
}

# Perform grid search with cross-validation to find the best parameters.
xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

# Train the final model with the best parameters.
xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)

# Evaluate the final model using cross-validation.
cv_results = cross_validate(xgboost_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
print(f"XGBoost Accuracy: {cv_results['test_accuracy'].mean()}")
print(f"XGBoost F1: {cv_results['test_f1'].mean()}")
print(f"XGBoost ROC AUC: {cv_results['test_roc_auc'].mean()}")

################################################
# LightGBM Hyperparameter Tuning
################################################

# Initialize a LightGBM classifier.
lgbm_model = LGBMClassifier(random_state=17)

# Define the parameter grid for LightGBM.
lgbm_params = {
    "learning_rate": [0.01, 0.1, 0.001],
    "n_estimators": [100, 300, 500, 1000],
    "colsample_bytree": [0.5, 0.7, 1]
}

# Perform grid search with cross-validation to find the best parameters.
lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

# Train the final model with the best parameters.
lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

# Evaluate the final model using cross-validation.
cv_results = cross_validate(lgbm_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
print(f"LightGBM Accuracy: {cv_results['test_accuracy'].mean()}")
print(f"LightGBM F1: {cv_results['test_f1'].mean()}")
print(f"LightGBM ROC AUC: {cv_results['test_roc_auc'].mean()}")

################################################
# CatBoost Optimization
################################################

# Initialize the CatBoost classifier with specific random state for reproducibility and silent mode to reduce log noise.
catboost_model = CatBoostClassifier(random_state=17, verbose=False)

# Define the hyperparameter grid for the CatBoost model.
catboost_params = {
    "iterations": [200, 500],  # Number of trees
    "learning_rate": [0.01, 0.1],  # Step size for tree's weight adjustment
    "depth": [3, 6]  # Depth of trees
}

# Perform grid search to find the best hyperparameters within the defined grid, using cross-validation.
catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

# Apply the best hyperparameters to the CatBoost model.
catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)

# Evaluate the optimized CatBoost model using cross-validation and several metrics.
cv_results = cross_validate(catboost_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

# Print out the average scores from cross-validation to assess model performance.
print(f"CatBoost Accuracy: {cv_results['test_accuracy'].mean()}")
print(f"CatBoost F1 Score: {cv_results['test_f1'].mean()}")
print(f"CatBoost ROC AUC: {cv_results['test_roc_auc'].mean()}")


################################################
# Feature Importance Analysis
################################################

# Function to plot the importance of features for the given model.
def plot_importance(model, features, num=len(X), save=False):
    # Create a DataFrame with feature importances.
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    # Plotting the feature importances.
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title('Feature Importance')
    plt.tight_layout()
    # Optionally save the plot.
    if save:
        plt.savefig('importances.png')


# Plot feature importance for all final models to understand which features are most influential in predicting churn.
plot_importance(rf_final, X)
plot_importance(xgboost_final, X)
plot_importance(lgbm_final, X)
plot_importance(catboost_final, X)

