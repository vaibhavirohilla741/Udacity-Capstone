from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Dataset

def clean_data(data):
    df = data.to_pandas_dataframe().dropna()
    df['BMI'] = df['BMI'].replace(0,df['BMI'].mean())
    df['BloodPressure'] = df['BloodPressure'].replace(0,df['BloodPressure'].mean())
    df['Glucose'] = df['Glucose'].replace(0,df['Glucose'].mean())
    df['Insulin'] = df['Insulin'].replace(0,df['Insulin'].mean())
    df['SkinThickness'] = df['SkinThickness'].replace(0,df['SkinThickness'].mean())
    q = df['Pregnancies'].quantile(0.98)
    data_cleaned = df[df['Pregnancies']<q]
    q = data_cleaned['BMI'].quantile(0.99)
    data_cleaned  = data_cleaned[data_cleaned['BMI']<q]
    q = data_cleaned['SkinThickness'].quantile(0.99)
    q = data_cleaned['Insulin'].quantile(0.95)
    data_cleaned  = data_cleaned[data_cleaned['Insulin']<q]
    q = data_cleaned['DiabetesPedigreeFunction'].quantile(0.99)
    data_cleaned  = data_cleaned[data_cleaned['DiabetesPedigreeFunction']<q]
    q = data_cleaned['Age'].quantile(0.99)
    data_cleaned  = data_cleaned[data_cleaned['Age']<q]
    X = df.drop(columns = ['Outcome'])
    y = df['Outcome']
    scalar = StandardScaler()
    X_scaled = scalar.fit_transform(X)
    return X_scaled,y
def main():
    parser = argparse.ArgumentParser()
    path='https://raw.githubusercontent.com/maheshcheetirala/Azure-Machine-Learning-ND-capstone/Main/diabetes.csv'
    ds = Dataset.Tabular.from_delimited_files(path=path)
    run = Run.get_context()
    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")
    args = parser.parse_args()
    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))
    x,y=clean_data(ds)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)
    joblib.dump(model,'outputs/model.joblib')
    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
if __name__ == '__main__':
    main()



