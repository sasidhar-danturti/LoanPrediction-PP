from sklearn.preprocessing import Imputer
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def showNulls(df):
    print df.isnull()

def countMissingCells(df):
    return sum(df.isnull().values.ravel())

def countMissingRows(df):
    return sum([True for idx,row in df.iterrows() if any(row.isnull())])

def fillMissingNumericValues(data,method='mean'):
    # since imputer works only with the numeric data, identify only numeric features
    numeric_data= data.select_dtypes(exclude=['object'])
    categoric_data=data.select_dtypes(include=['object'])

    imputer = Imputer(strategy=method)
    imputer.fit(numeric_data)
    imputed_data =pd.DataFrame(imputer.transform(numeric_data),columns=numeric_data.columns.values)

    return pd.concat([categoric_data,imputed_data],axis=1)

def fillMissingCategoricValues(data,exclude_columns,method='mode'):
    # since imputer works only with the numeric data, identify only numeric features
    numeric_data= data.select_dtypes(exclude=['object'])
    categoric_data=data.select_dtypes(include=['object'])


    for category in categoric_data.columns:
        if (category not in exclude_columns):
            categoric_data[category].fillna(categoric_data[category].mode()[0], inplace=True)
            categoric_data[category]=LabelEncoder().fit(categoric_data[category]).transform(categoric_data[category])

    return pd.concat([categoric_data,numeric_data],axis=1)


def dropFeatures(data,data_type):
    # since imputer works only with the numeric data, identify only numeric features
    return data.select_dtypes(exclude=['object'])

