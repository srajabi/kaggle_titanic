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
import pandas as pd
# fix random seed for reproducibility
numpy.random.seed(7)



titanic_file_path = './titanic/train.csv'

titanic_data = pandas.read_csv(titanic_file_path)
titanic_data = titanic_data.fillna(method='ffill')

# PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
features = [
        'Pclass','Sex','Age', 'SibSp', 'Parch', 'Fare', 'Embarked'
]

def get_binned_ages(data_frame):
       bins = [0, 16, 32, 48, 64, 150]
       labels = [0, 1, 2, 3, 4]
       data_frame['Age'] = pandas.cut(data_frame['Age'], bins=bins, labels=labels)
       return data_frame


titanic_data = get_binned_ages(titanic_data)

X = titanic_data[features]
y = titanic_data.Survived

cat_columns = ["Sex", "Embarked", 'Age']
df_processed = pd.get_dummies(X, prefix_sep="__",
                              columns=cat_columns)
lengthOfFeatures = len(df_processed.columns)

train_X, val_X, train_y, val_y = train_test_split(df_processed, y, random_state=1)

print(lengthOfFeatures)

model = Sequential()
model.add(Dense(16, input_dim=lengthOfFeatures, activation='relu'))
model.add(Dense(14, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(train_X, train_y, epochs=150, batch_size=150)



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
