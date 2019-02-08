import pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error

# TESTING AND MODEL CREATION
titanic_file_path = './titanic/train.csv'

titanic_data = pandas.read_csv(titanic_file_path)
titanic_data = titanic_data.fillna(method='ffill')

# todo one-hot encode Sex
features = [
    'Age', 'Pclass'
]


X = titanic_data[features]
y = titanic_data.Survived

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


rf_model = RandomForestClassifier(random_state=1)
rf_model.fit(train_X, train_y)
y_predicted = rf_model.predict(val_X)

score = rf_model.score(train_X, train_y)
print(score)
print(y_predicted)
print(val_y)


# SUBMISSION
test_file_path = './titanic/test.csv'

test_data = pandas.read_csv(test_file_path)
test_data = test_data.fillna(method='ffill')

X_submit = test_data[features]
rf_model.fit(X, y) # use entire dataset for training

test_predictions = rf_model.predict(X_submit)

output = pandas.DataFrame({'PassengerId' : test_data.PassengerId,
                           'Survived' : test_predictions})

output.to_csv('submission.csv', index=False)