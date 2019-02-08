import pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error

titanic_file_path = './titanic/train.csv'

titanic_data = pandas.read_csv(titanic_file_path)

# todo one-hot encode Sex
features = [
    'Age', 'Pclass'
]

titanic_data = titanic_data.fillna(method='ffill')

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
