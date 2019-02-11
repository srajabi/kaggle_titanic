import pandas
import numpy
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score


class TitanicModel(object):

    def __init__(self):
        titanic_file_path = './titanic/train.csv'
        self.train_df = pandas.read_csv(titanic_file_path)

        test_file_path = './titanic/test.csv'
        self.test_df = pandas.read_csv(test_file_path)

    def create_model(self):
        return RandomForestClassifier(n_estimators=200,
                                      min_samples_leaf=3,
                                      max_features=0.5,
                                      n_jobs=-1)

    def fillna_method(self, data_frame):
        return data_frame.fillna(method='ffill')

    def drop_useless(self, data_frame):
        return data_frame.drop(['Ticket', 'Cabin'], axis=1)

    def create_title(self, data_frame):
        data_frame['Title'] = data_frame.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

        print(data_frame['Title'].unique())

        title_map = {
            'Mr': 1,
            'Miss': 2,
            'Mlle': 2,
            'Ms': 2,
            'Mrs': 3,
            'Mme': 3,
            'Master': 4,
            'Lady': 3,
            'Countess': 3,
            'Capt': 1,
            'Col': 1,
            'Don': 1,
            'Dr': 1,
            'Major': 1,
            'Rev': 1,
            'Sir': 1,
            'Jonkheer': 1,
            'Dona': 2
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
        data_frame['Age'] = data_frame['Age'].ffill()
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
        bins = [0, 8, 15, 31, 64, 128, 256, 1024]
        labels = [0, 1, 2, 3, 4, 5, 6]
        data_frame['Fare'] = pandas.cut(data_frame['Fare'], bins=bins, labels=labels).astype(int)
        return data_frame

    def create_family_size(self, data_frame):
        data_frame['FamilySize'] = data_frame['Parch'] + data_frame['SibSp'] + 1
        data_frame['Singleton'] = data_frame['FamilySize'].map(lambda s: 1 if s == 1 else 0)
        data_frame['SmallFamily'] = data_frame['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
        data_frame['LargeFamily'] = data_frame['FamilySize'].map(lambda s: 1 if 5 <= s else 0)

        data_frame.drop('Parch', axis=1, inplace=True)
        data_frame.drop('SibSp', axis=1, inplace=True)
        return data_frame

    def process_embarked(self, data_frame):
        data_frame['Embarked'].fillna('S', inplace=True)
        dummies = pandas.get_dummies(data_frame['Embarked'], prefix='Embarked')
        data_frame = pandas.concat([data_frame, dummies], axis=1)
        data_frame.drop('Embarked', axis=1, inplace=True)
        return data_frame

    def process_cabin(self, data_frame):
        data_frame['Cabin'].fillna('T', inplace=True)
        data_frame['Cabin'] = data_frame['Cabin'].map(lambda c: c[0])
        dummies = pandas.get_dummies(data_frame['Cabin'], prefix='Cabin')
        data_frame = pandas.concat([data_frame, dummies], axis=1)
        data_frame.drop('Cabin', axis=1, inplace=True)
        return data_frame

    def fill_age(self, row, grouped_median_train):
        condition = (
            (grouped_median_train['Sex'] == row['Sex']) &
            (grouped_median_train['Title'] == row['Title']) &
            (grouped_median_train['Pclass'] == row['Pclass'])
        )
        if numpy.isnan(grouped_median_train[condition]['Age'].values[0]):
            condition = (
                (grouped_median_train['Sex'] == row['Sex']) &
                (grouped_median_train['Pclass'] == row['Pclass'])
            )

        return grouped_median_train[condition]['Age'].values[0]

    def get_titles(self, data_frame):

        title_dict = {
            "Capt": "Mr",
            "Col": "Mr",
            "Major": "Mr",
            "Jonkheer": "Mr",
            "Don": "Mr",
            "Dona": "Mrs",
            "Sir": "Mr",
            "Dr": "Mr",
            "Rev": "Mr",
            "the Countess": "Mrs",
            "Countess": "Mrs",
            "Mme": "Mrs",
            "Mlle": "Miss",
            "Ms": "Miss",
            "Mr": "Mr",
            "Mrs": "Mrs",
            "Miss": "Miss",
            "Master": "Master",
            "Lady" : "Mrs"
        }

        data_frame['Title'] = data_frame['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

        data_frame['Title'] = data_frame['Title'].map(title_dict)

        return data_frame

    def process_names(self, data_frame):
        data_frame.drop('Name', axis=1, inplace=True)

        titles_dummies = pandas.get_dummies(data_frame['Title'], prefix='Title')
        data_frame = pandas.concat([data_frame, titles_dummies], axis=1)

        data_frame.drop('Title', axis=1, inplace=True)

        return data_frame

    def fill_age(self, row, grouped_median_train):
        condition = (
            (grouped_median_train['Sex'] == row['Sex']) &
            (grouped_median_train['Title'] == row['Title']) &
            (grouped_median_train['Pclass'] == row['Pclass'])
        )
        if numpy.isnan(grouped_median_train[condition]['Age'].values[0]):
            condition = (
                (grouped_median_train['Sex'] == row['Sex']) &
                (grouped_median_train['Pclass'] == row['Pclass'])
            )

        return grouped_median_train[condition]['Age'].values[0]

    def process_age(self, data_frame):
        grouped_train = data_frame.groupby(['Sex', 'Pclass', 'Title'])
        grouped_median_train = grouped_train.median()
        grouped_median_train = grouped_median_train.reset_index()
        grouped_median_train = grouped_median_train[['Sex', 'Pclass', 'Title', 'Age']]

        data_frame['Age'] = data_frame.apply(lambda row: self.fill_age(row, grouped_median_train) if numpy.isnan(row['Age']) else row['Age'], axis=1)

        return data_frame

    def fill_fare(self, row, grouped_median_train):
        condition = (
            (grouped_median_train['Sex'] == row['Sex']) &
            (grouped_median_train['Title'] == row['Title']) &
            (grouped_median_train['Pclass'] == row['Pclass'])
        )
        if numpy.isnan(grouped_median_train[condition]['Fare'].values[0]):
            condition = (
                (grouped_median_train['Sex'] == row['Sex']) &
                (grouped_median_train['Pclass'] == row['Pclass'])
            )

        return grouped_median_train[condition]['Fare'].values[0]

    def process_fare(self, data_frame):
        grouped_train = data_frame.groupby(['Sex', 'Pclass', 'Title'])
        grouped_median_train = grouped_train.median()
        grouped_median_train = grouped_median_train.reset_index()
        grouped_median_train = grouped_median_train[['Sex', 'Pclass', 'Title', 'Fare']]

        data_frame['Fare'] = data_frame.apply(lambda row: self.fill_fare(row, grouped_median_train) if numpy.isnan(row['Fare']) else row['Fare'], axis=1)

        return data_frame

    def process_data(self, df):
        #train = self.fillna_method(train)
        #train = self.drop_useless(train)
        #rain = self.drop_passenger_id(train)
        #train = self.get_binned_ages(train)
        #train = self.is_alone(train)
        #train = self.fill_embarked(train)
        #train = self.fare_binning(train)
        #train = self.create_title(train)
        #train['Title'] = self.one_hot_encode(train['Title'])

        df = self.sex_to_categorical(df)
        df = self.create_family_size(df)
        df = self.process_embarked(df)
        df = self.process_cabin(df)
        df = self.get_titles(df)
        df = self.process_age(df)
        df = self.process_fare(df)
        df = self.process_names(df)

        df.drop('Ticket', axis=1, inplace=True)

        return df


    def test_model(self):
        train = self.process_data(self.train_df)

        model = self.create_model()

        y = train['Survived']
        train.drop('Survived', axis=1, inplace=True)
        X = train

        train_X, validate_X, train_y, validate_y = train_test_split(X, y, random_state=9)

        model.fit(train_X, train_y)
        print(model.score(train_X, train_y))
        y_predicted = model.predict(validate_X)
        score = accuracy_score(validate_y, y_predicted)
        print('Mean Accuracy: {}'.format(score))

    def submit_model(self):
        train = self.process_data(self.train_df)
        #train.drop('Title_Royalty', axis=1, inplace=True)
        test = self.process_data(self.test_df)

        model = self.create_model()

        y = train['Survived']
        train.drop('Survived', axis=1, inplace=True)
        X = train

        model.fit(X, y)
        print(model.score(X, y))

        #print(test.columns.values)
        #print(train.columns.values)

        test_predictions = model.predict(test)

        output = pandas.DataFrame({'PassengerId': test.PassengerId,
                                   'Survived': test_predictions})

        output.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    titanic_model = TitanicModel()
    titanic_model.test_model()
    titanic_model.submit_model()
