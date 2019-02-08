import pandas
import numpy
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class TitanicModel(object):

    def __init__(self):
        titanic_file_path = './titanic/train.csv'
        self.train_df = pandas.read_csv(titanic_file_path)

        test_file_path = './titanic/test.csv'
        self.test_df = pandas.read_csv(test_file_path)

    def create_model(self):
        return RandomForestClassifier(random_state=1,
                                      n_estimators=10,
                                      max_features='auto')

    def fillna_method(self, data_frame):
        return data_frame.fillna(method='ffill')

    def drop_useless(self, data_frame):
        return data_frame.drop(['Ticket', 'Cabin'], axis=1)

    def create_title(self, data_frame):
        data_frame['Title'] = data_frame.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

        title_map = {
            'Mr': 1,
            'Miss': 2,
            'Mlle': 2,
            'Ms': 2,
            'Mrs': 3,
            'Mme': 3,
            'Master': 4,
            'Lady': 5,
            'Countess': 5,
            'Capt': 5,
            'Col': 5,
            'Don': 5,
            'Dr': 5,
            'Major': 5,
            'Rev': 5,
            'Sir': 5,
            'Jonkheer': 5,
            'Dona': 5
        }
        data_frame = data_frame.replace({'Title': title_map})
        data_frame['Title'] = data_frame['Title'].fillna(0)

        return data_frame.drop(['Name'], axis=1)

    def drop_passenger_id(self, data_frame):
        return data_frame.drop(['PassengerId'], axis=1)

    def sex_to_categorical(self, data_frame):
        sex_mapping = {'female': 1, 'male': 0}
        data_frame = data_frame.replace({'Sex': sex_mapping})
        return data_frame

    def one_hot_encode(self, column):
        column = array(column)
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(column)
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        return onehot_encoder.fit_transform(integer_encoded)

    def get_binned_ages(self, data_frame):
        bins = [0, 16, 32, 48, 64, 150]
        labels = [0, 1, 2, 3, 4]
        data_frame['Age'] = pandas.cut(data_frame['Age'], bins=bins, labels=labels)
        return data_frame

    def is_alone(self, data_frame):
        data_frame['IsAlone'] = data_frame.apply(
            lambda row: int((row['SibSp'] + row['Parch']) == 0), axis=1)
        return data_frame

    def fill_embarked(self, data_frame):
        freq_port = data_frame.Embarked.dropna().mode()[0]

        data_frame['Embarked'] = data_frame['Embarked'].fillna(freq_port)
        data_frame['Embarked'] = data_frame['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

        return data_frame

    def fare_binning(self, data_frame):
        bins = [0, 8, 15, 31, 600]
        labels = [0, 1, 2, 3]
        data_frame['Fare'] = pandas.cut(data_frame['Fare'], bins=bins, labels=labels).astype(int)
        return data_frame

    def test_model(self):
        train = self.fillna_method(self.train_df)
        train = self.drop_useless(train)
        train = self.drop_passenger_id(train)
        train = self.sex_to_categorical(train)
        train = self.create_title(train)
        train = self.get_binned_ages(train)
        train = self.is_alone(train)
        train = self.fill_embarked(train)
        train = self.fare_binning(train)

        model = self.create_model()

        X = train[['Pclass', 'Sex', 'Fare', 'Title', 'Age', 'IsAlone', 'Embarked']]
        y = train['Survived']

        train_X, validate_X, train_y, validate_y = train_test_split(X, y, random_state=1)

        model.fit(train_X, train_y)
        y_predicted = model.predict(validate_X)
        score = model.score(validate_X, validate_y)
        print('Mean Accuracy: {}'.format(score))


if __name__ == "__main__":
    titanic_model = TitanicModel()
    titanic_model.test_model()
