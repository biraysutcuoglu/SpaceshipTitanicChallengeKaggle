#find which passengers transported to the alternate dimension
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
# from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

train_dataFull = pd.read_csv('train.csv', index_col='PassengerId')
test_dataFull = pd.read_csv('test.csv', index_col='PassengerId')

#encode for home planet(3), cryosleep(3), destination(3), VIP(2-boolean),
train_data = train_dataFull.drop(['Cabin', 'Name'], axis=1)
test_data = test_dataFull.drop(['Cabin', 'Name'], axis=1)

y = train_data.Transported
X_train = train_data.drop(['Transported'], axis=1)
features = X_train.columns

objectCols = list(X_train.select_dtypes(['object']).columns)

# # print(objectCols)
# # check cols that can be encoded-less than 15 unique values
# for col in objectCols:
#     print(train_dataFull[col].value_counts())

# #check number of na values in the cols
# for col in features:
#     print(col + ': ' + str(X_train[col].isna().sum()))

#imputation for Age, RoomService, FoodCourt, Shopping Mall, Spa, VRDeck - numeric cols
colsMissingNumeric = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# print(X_train[colsMissing])
myImputerNumeric = SimpleImputer(strategy='mean')
imputed_X_train_numeric = pd.DataFrame(myImputerNumeric.fit_transform(X_train[colsMissingNumeric]))
imputed_test_numeric = pd.DataFrame(myImputerNumeric.transform(test_data[colsMissingNumeric]))

print(imputed_X_train_numeric.columns)

#name imputed cols
imputed_X_train_numeric.columns = X_train[colsMissingNumeric].columns
imputed_test_numeric.columns = test_data[colsMissingNumeric].columns

imputed_X_train_numeric.index = X_train.index
imputed_test_numeric.index = test_data.index

#check number of empty valued cols after imputation
# for col in imputed_X_train.columns:
#     print(col + ': ' + str(imputed_X_train[col].isna().sum()))

#impute object cols before encoding
myImputerObject = SimpleImputer(strategy='most_frequent')
imputed_X_train_object = pd.DataFrame(myImputerObject.fit_transform(X_train[objectCols]))
imputed_test_object = pd.DataFrame(myImputerObject.transform(test_data[objectCols]))

#name imputed cols
imputed_X_train_object.columns = X_train[objectCols].columns
imputed_test_object.columns = test_data[objectCols].columns

#encode imputed object cols with one hot encoding
#these cols are not suitable for ordering and have low cardinality
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(imputed_X_train_object))
OH_cols_test = pd.DataFrame(OH_encoder.transform(imputed_test_object))

OH_cols_train.index = X_train.index
OH_cols_test.index = test_data.index

OH_X_train = pd.concat([imputed_X_train_numeric,OH_cols_train], axis=1)
OH_X_valid = pd.concat([imputed_test_numeric,OH_cols_test], axis=1)

titanic_model = RandomForestRegressor(n_estimators = 100, random_state = 0)
titanic_model.fit(OH_X_train, y)
predictions = titanic_model.predict(OH_X_valid)
predictions = np.rint(predictions)
bool_predictions = []

for i in predictions:
    if i == 1:
        bool_predictions.append(True)
    else:
        bool_predictions.append(False)

# print(bool_predictions)
output = {'PassengerId': OH_X_valid.index,
                       'Transported': bool_predictions}

df = pd.DataFrame(output)
# print(df.info())
df.to_csv('submission.csv',index=False)
# # #XGB
# titanic_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
# # titanic_model.fit(OH_X_train, y)
# # # predictions = titanic_model.predict(OH_X_valid)




















