import pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# TESTING AND MODEL CREATION
titanic_file_path = './titanic/train.csv'

titanic_data = pandas.read_csv(titanic_file_path)
titanic_data = titanic_data.fillna(method='ffill')

sexes = array(titanic_data['Sex'])

# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(sexes)

# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

# todo one-hot encode Sex
features = [
    'Age', 'Pclass', 'Sex', 'Parch', 'SibSp'
]


X = titanic_data[features]
X.Sex = onehot_encoded
y = titanic_data.Survived

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

rf_model = RandomForestClassifier(random_state=1)
rf_model.fit(train_X, train_y)
y_predicted = rf_model.predict(val_X)

score = rf_model.score(val_X, val_y)
print('Mean Accuracy: {}'.format(score))

# SUBMISSION
'''
test_file_path = './titanic/test.csv'

test_data = pandas.read_csv(test_file_path)
test_data = test_data.fillna(method='ffill')

X_submit = test_data[features]
rf_model.fit(X, y) # use entire dataset for training

test_predictions = rf_model.predict(X_submit)

output = pandas.DataFrame({'PassengerId' : test_data.PassengerId,
                           'Survived' : test_predictions})

output.to_csv('submission.csv', index=False)
'''
