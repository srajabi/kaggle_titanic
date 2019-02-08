import pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from numpy import array

from keras.models import Sequential
from keras.layers import Dense
import numpy

# fix random seed for reproducibility
numpy.random.seed(7)



titanic_file_path = './titanic/train.csv'

titanic_data = pandas.read_csv(titanic_file_path)
titanic_data = titanic_data.fillna(method='ffill')

# todo one-hot encode Sex
features = [
    'Pclass','Sex','SibSp','Parch','Embarked'
]

X = titanic_data[features]
y = titanic_data.Survived


lengthOfFeatures = len(features)

#Auto encodes any dataframe column of type category or object.
def dummyEncode(df):
        columnsToEncode = list(df.select_dtypes(include=['category','object']))
        le = LabelEncoder()
        for feature in columnsToEncode:
            try:
                df[feature] = le.fit_transform(df[feature])
            except:
                print('Error encoding '+feature)
        return df



dummyEncoding =  dummyEncode(X)


train_X, val_X, train_y, val_y = train_test_split(dummyEncoding, y, random_state=1)


model = Sequential()
model.add(Dense(10, input_dim=lengthOfFeatures, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(train_X, train_y, epochs=150, batch_size=10)



predictions = model.predict(val_X)
# round predictions
rounded = [round(x[0]) for x in predictions]


correct = 0
for x, y in map(None, rounded, val_y):
    if x == y:
       correct  = correct + 1

print('total: ' + str(len(predictions)))
print('correct: ' + str(correct))
acc = correct / float(len(predictions))
print('acc' + str(acc))
