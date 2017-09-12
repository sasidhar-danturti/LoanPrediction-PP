import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import NullHandler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


target="Loan_Status"
data = pd.read_csv("Data/train.csv").drop("Loan_ID",axis=1)

y = data[target]

# NullHandler.showNulls(data)

# print (NullHandler.countMissingCells(data))
#
# print (NullHandler.countMissingRows(data))


print(data.info)

data = NullHandler.dropFeatures(data,"object")
print data.shape
data_imputed= NullHandler.fillMissingNumericValues(data,method='median')
data_imputed= NullHandler.fillMissingCategoricValues(data_imputed,exclude_columns=['Loan_ID'])

print (NullHandler.countMissingCells(data_imputed))

print (NullHandler.countMissingRows(data_imputed))


X_train, X_test, y_train, y_test = train_test_split(data_imputed, y, test_size=0.2, random_state=10)

base_model = GaussianNB().fit(X_train,y_train)


train_predictions= base_model.predict(X_train)
test_predictions =base_model.predict(X_test)

print(accuracy_score(y_train,train_predictions))


print(accuracy_score(y_test,test_predictions))

# #tar
# # print data_imputed
#

# # print(data.head())
#
# # print(data.info())
# #
# print(data.describe())
# print data.info()
#
# print("-----------------------------------")
# # print(data['Loan_Status'].value_counts())
# #
# # print(data['Property_Area'].value_counts())
# #
#
# # print(data['Education'].value_counts())
# print(data['LoanAmount'].head())
# data['LoanAmount'].fillna((data['LoanAmount'].mean()), inplace=True)
# print(data['LoanAmount'].head())
# data['Gender'].fillna("Female", inplace=True)
#
# print(data.info())
#
# # data.plot.hist()
#
# # data.hist(bins=4)
# data['LoanAmount'].value_counts().plot(kind='bar')
# plt.show()
#
#
