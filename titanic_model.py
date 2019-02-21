import pandas
import numpy
from numpy import array
import seaborn
import matplotlib
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve


class TitanicModel(object):

    def __init__(self):
        titanic_file_path = './titanic/train.csv'
        self.train_df = pandas.read_csv(titanic_file_path)

        test_file_path = './titanic/test.csv'
        self.test_df = pandas.read_csv(test_file_path)

    @staticmethod
    def create_nn_model():
        number_of_features = 23
        model = Sequential()
        model.add(Dense(number_of_features, input_dim=number_of_features, activation='relu'))
        model.add(Dense(number_of_features // 2, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
        return model

    def create_model(self):
        return RandomForestClassifier(n_estimators=200,
                                      min_samples_leaf=3,
                                      max_features=0.5,
                                      n_jobs=-1,
                                      random_state=76)

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
        data_frame['SmallFamily'] = data_frame['FamilySize'].map(lambda s: 1 if s == 2 else 0)
        data_frame['MediumFamily'] = data_frame['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
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

        data_frame["Fare"] = data_frame["Fare"].map(lambda i: numpy.log(i) if i > 0 else 0)

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
        #df.drop('PassengerId', axis=1, inplace=True)

        #print(df.columns.values)

        return df

    def create_classifiers(self):
        random_state = 2
        classifiers = []
        classifiers.append(SVC(random_state=random_state))
        classifiers.append(DecisionTreeClassifier(random_state=random_state))
        classifiers.append(
            AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state), random_state=random_state,
                               learning_rate=0.1))
        classifiers.append(RandomForestClassifier(random_state=random_state))
        classifiers.append(ExtraTreesClassifier(random_state=random_state))
        classifiers.append(GradientBoostingClassifier(random_state=random_state))
        classifiers.append(MLPClassifier(random_state=random_state))
        classifiers.append(KNeighborsClassifier())
        classifiers.append(LogisticRegression(random_state=random_state))
        classifiers.append(LinearDiscriminantAnalysis())

        return classifiers

    def test_classifiers(self):
        train = self.process_data(self.train_df)
        y = train['Survived']
        train.drop('Survived', axis=1, inplace=True)
        X = train

        classifiers = self.create_classifiers()

        kfold = StratifiedKFold(n_splits=10)
        cv_results = []
        for classifier in classifiers:
            cv_results.append(cross_val_score(classifier, X, y=y, scoring="accuracy", cv=kfold, n_jobs=4))

        cv_means = []
        cv_std = []
        for cv_result in cv_results:
            cv_means.append(cv_result.mean())
            cv_std.append(cv_result.std())

        cv_res = pandas.DataFrame(
            {"CrossValMeans": cv_means, "CrossValerrors": cv_std, "Algorithm": ["SVC", "DecisionTree", "AdaBoost",
                                                                                "RandomForest", "ExtraTrees",
                                                                                "GradientBoosting",
                                                                                "MultipleLayerPerceptron",
                                                                                "KNeighboors", "LogisticRegression",
                                                                                "LinearDiscriminantAnalysis"]})

        g = seaborn.barplot("CrossValMeans", "Algorithm", data=cv_res, palette="Set3", orient="h", **{'xerr': cv_std})
        g.set_xlabel("Mean Accuracy")
        g = g.set_title("Cross validation scores")

        pyplot.savefig('testfigure.png', bbox_inches='tight')

    def hp_tune_ada(self):
        train = self.process_data(self.train_df)
        y = train['Survived']
        train.drop('Survived', axis=1, inplace=True)
        X = train

        kfold = StratifiedKFold(n_splits=10)

        ### META MODELING  WITH ADABOOST, RF, EXTRATREES and GRADIENTBOOSTING

        # Adaboost
        DTC = DecisionTreeClassifier()

        adaDTC = AdaBoostClassifier(DTC, random_state=4)

        ada_param_grid = {"base_estimator__criterion": ["gini", "entropy"],
                          "base_estimator__splitter": ["best", "random"],
                          "algorithm": ["SAMME", "SAMME.R"],
                          "n_estimators": [1, 2],
                          "learning_rate": [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 1.5]}

        gsadaDTC = GridSearchCV(adaDTC, param_grid=ada_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)

        gsadaDTC.fit(X, y)

        ada_best = gsadaDTC.best_estimator_
        print(ada_best)
        print(gsadaDTC.best_score_)

    def hp_tune_xtra(self):
        train = self.process_data(self.train_df)
        y = train['Survived']
        train.drop('Survived', axis=1, inplace=True)
        X = train

        kfold = StratifiedKFold(n_splits=10)

        # ExtraTrees
        ExtC = ExtraTreesClassifier()

        ## Search grid for optimal parameters
        ex_param_grid = {"max_depth": [None],
                         "max_features": [1, 3, 10],
                         "min_samples_split": [2, 3, 10],
                         "min_samples_leaf": [1, 3, 10],
                         "bootstrap": [False],
                         "n_estimators": [100, 300],
                         "criterion": ["gini"]}

        gsExtC = GridSearchCV(ExtC, param_grid=ex_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)

        gsExtC.fit(X, y)

        ExtC_best = gsExtC.best_estimator_

        # Best score
        gsExtC.best_score_

        print(ExtC_best)
        print(gsExtC.best_score_)


    def hp_tune_rf(self):
        train = self.process_data(self.train_df)
        y = train['Survived']
        train.drop('Survived', axis=1, inplace=True)
        X = train

        kfold = StratifiedKFold(n_splits=10)

        # RFC Parameters tunning
        RFC = RandomForestClassifier()

        ## Search grid for optimal parameters
        rf_param_grid = {"max_depth": [None],
                         "max_features": [1, 3, 10],
                         "min_samples_split": [2, 3, 10],
                         "min_samples_leaf": [1, 3, 10],
                         "bootstrap": [False],
                         "n_estimators": [100, 300],
                         "criterion": ["gini"]}

        gsRFC = GridSearchCV(RFC, param_grid=rf_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)

        gsRFC.fit(X, y)

        RFC_best = gsRFC.best_estimator_

        # Best score
        gsRFC.best_score_

        print(RFC_best)
        print(gsRFC.best_score_)

    def hp_tune_gb(self):
        train = self.process_data(self.train_df)
        y = train['Survived']
        train.drop('Survived', axis=1, inplace=True)
        X = train

        kfold = StratifiedKFold(n_splits=10)

        # Gradient boosting tunning

        GBC = GradientBoostingClassifier()
        gb_param_grid = {'loss': ["deviance"],
                         'n_estimators': [100, 200, 300],
                         'learning_rate': [0.1, 0.05, 0.01],
                         'max_depth': [4, 8],
                         'min_samples_leaf': [100, 150],
                         'max_features': [0.3, 0.1]
                         }

        gsGBC = GridSearchCV(GBC, param_grid=gb_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)

        gsGBC.fit(X, y)

        GBC_best = gsGBC.best_estimator_

        # Best score
        gsGBC.best_score_

        print(GBC_best)
        print(gsGBC.best_score_)

    def hp_tune_svc(self):
        train = self.process_data(self.train_df)
        y = train['Survived']
        train.drop('Survived', axis=1, inplace=True)
        X = train

        kfold = StratifiedKFold(n_splits=10)

        ### SVC classifier
        SVMC = SVC(probability=True)
        svc_param_grid = {'kernel': ['rbf'],
                          'gamma': [0.001, 0.01, 0.1, 1],
                          'C': [1, 10, 50, 100, 200, 300, 1000]}

        gsSVMC = GridSearchCV(SVMC, param_grid=svc_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)

        gsSVMC.fit(X, y)

        SVMC_best = gsSVMC.best_estimator_

        # Best score
        gsSVMC.best_score_

        print(SVMC_best)
        print(gsSVMC.best_score_)

    def hp_tune_lda(self):
        train = self.process_data(self.train_df)
        y = train['Survived']
        train.drop('Survived', axis=1, inplace=True)
        X = train

        kfold = StratifiedKFold(n_splits=10)

        lda = LinearDiscriminantAnalysis()
        param_grid = {'store_covariance': [True, False],
                      'tol': [0.00001, 0.0001, 0.001, 0.01, 0.1]}

        gsLDA = GridSearchCV(lda, param_grid=param_grid, cv=kfold, scoring='accuracy', n_jobs=4, verbose=1)

        gsLDA.fit(X, y)

        best = gsLDA.best_estimator_
        best_score = gsLDA.best_score_

        print(best)
        print(best_score)

    def create_ensemble(self):
        train = self.process_data(self.train_df)
        y = train['Survived']
        train.drop('Survived', axis=1, inplace=True)
        train.drop('PassengerId', axis=1, inplace=True)
        X = train

        dtc = DecisionTreeClassifier(criterion='gini', splitter='random')
        ada = AdaBoostClassifier(dtc, algorithm='SAMME', learning_rate=0.0001, n_estimators=1, random_state=4)

        xtra = ExtraTreesClassifier(max_depth=None, max_features=10, min_samples_split=2, min_samples_leaf=10, bootstrap=False, n_estimators=300, criterion='gini')

        rfc = RandomForestClassifier(max_depth=None, max_features=3, min_samples_split=10, min_samples_leaf=3, bootstrap=False, n_estimators=300, criterion='gini')

        gb = GradientBoostingClassifier(loss='deviance', n_estimators=200, learning_rate=0.05, max_depth=4, min_samples_leaf=100, max_features=0.3)

        lda = LinearDiscriminantAnalysis(store_covariance=True, tol=0.000001)

        votingC = VotingClassifier(estimators=[('Ada', ada), ('Xtra', xtra), ('RFC', rfc), ('GB', gb), ('LDA', lda)], voting='soft', n_jobs=4)

        votingC = votingC.fit(X, y)

        test = self.process_data(self.test_df)
        test.drop('PassengerId', axis=1, inplace=True)
        test_survived = pandas.Series(votingC.predict(test), name='Survived')

        result = pandas.concat([self.test_df['PassengerId'], test_survived], axis=1)
        result.to_csv('ensemble.csv', index=False)
        
    def test_nn_model(self):
        seed = 7
        numpy.random.seed(seed)

        train = self.process_data(self.train_df)
        y = train['Survived']
        train.drop('Survived', axis=1, inplace=True)
        train.drop('PassengerId', axis=1, inplace=True)
        X = train

        neural_network = KerasClassifier(build_fn=self.create_nn_model, epochs=150, batch_size=150, verbose=0)
        kfold = StratifiedKFold(n_splits=10)
        scores = cross_val_score(neural_network, X, y, scoring = "accuracy", cv = kfold, n_jobs=4)
        print("%.2f%% (std %.2f)" % (numpy.mean(scores), numpy.std(scores)))
        
    
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

        #print(classification_report(validate_y, y_predicted))
        #print(confusion_matrix(validate_y, y_predicted))

    def submit_nn_model(self):
        train = self.process_data(self.train_df)
        #train.drop('Title_Royalty', axis=1, inplace=True)
        test = self.process_data(self.test_df)

        y = train['Survived']
        train.drop('Survived', axis=1, inplace=True)
        X = train

        model = self.create_nn_model()

        model.fit(X, y, epochs=150, batch_size=150)

        model.fit(X, y)

        test_predictions = model.predict(test)

        rounded = [int(round(x)) for x in test_predictions]

        output = pandas.DataFrame({'PassengerId': test.PassengerId,
                                   'Survived': rounded})

        output.to_csv('submission.csv', index=False)
        
    def submit_model(self):
        train = self.process_data(self.train_df)
        #train.drop('Title_Royalty', axis=1, inplace=True)
        test = self.process_data(self.test_df)
        test.drop('PassengerId', axis=1, inplace=True)

        model = self.create_model()

        y = train['Survived']
        train.drop('Survived', axis=1, inplace=True)
        train.drop('PassengerId', axis=1, inplace=True)
        X = train

        model.fit(X, y)
        print(model.score(X, y))

        def rf_feat_importance(m, df):
            return pandas.DataFrame({'cols': df.columns, 'imp':
                m.feature_importances_}).sort_values('imp', ascending=False)

        feature_importance = rf_feat_importance(model, train)

        to_keep = feature_importance[feature_importance.imp > 0.01].cols

        X = train[to_keep]

        model.fit(X, y)

        scores = cross_val_score(model, X, y)
        print(scores)
        print(model.score(X, y))

        #print(test.columns.values)
        #print(train.columns.values)

        test_predictions = model.predict(test[to_keep])

        output = pandas.DataFrame({'PassengerId': self.test_df.PassengerId,
                                   'Survived': test_predictions})

        output.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    titanic_model = TitanicModel()
    #titanic_model.create_ensemble()
    #titanic_model.test_classifiers()
    #titanic_model.test_model()
    titanic_model.test_model()
