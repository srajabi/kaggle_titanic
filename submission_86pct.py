import pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
import numpy
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if big_string.find(substring) != -1:
            return substring
    print big_string
    return numpy.nan

def GetHotSex(Sex):
    Sex = array(Sex)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(Sex)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    return onehot_encoder.fit_transform(integer_encoded)

def GetBinnedAges(Ages):
    BinnedAges = numpy.empty_like(Ages)
    for i in range(len(Ages)):
        if Ages[i] < 12:
            BinnedAges[i] = 0
        elif Ages[i] < 60:
            BinnedAges[i] = 2
        else:
            BinnedAges[i] = 3

    return pandas.DataFrame(BinnedAges)

# TESTING AND MODEL CREATION
titanic_file_path = './titanic/train.csv'

titanic_data = pandas.read_csv(titanic_file_path)

titanic_data = titanic_data.fillna(method='ffill')

#creating a title column from name
title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
            'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
            'Don', 'Jonkheer']

titanic_data['Title']=titanic_data['Name'].map(lambda x: substrings_in_string(x, title_list))

#replacing all titles with mr, mrs, miss, master
def replace_titles(x):
    title=x['Title']
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Countess', 'Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title =='Dr':
        if x['Sex']=='Male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title
        
titanic_data['Title']=titanic_data.apply(replace_titles, axis=1)
HotTitle = GetHotSex(titanic_data['Title'])

HotSex = GetHotSex(titanic_data['Sex'])
HotParch = GetHotSex(titanic_data['Parch'])
BinnedAge = GetBinnedAges(titanic_data['Age'])

features = [
    'Pclass', 
    'Sex', 
    'Fare', 
    #'Age',
    'Title'
]

X = titanic_data[features]
X.Sex = HotSex
#X.Age = BinnedAge
X.Title = HotTitle
y = titanic_data.Survived

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

rf_model = RandomForestClassifier(random_state=1, n_estimators=10, max_features="auto")
rf_model.fit(train_X, train_y)
y_predicted = rf_model.predict(val_X)

score = rf_model.score(val_X, val_y)
print('Mean Accuracy: {}'.format(score))

# SUBMISSION
test_file_path = './titanic/test.csv'

test_data = pandas.read_csv(test_file_path)
test_data = test_data.fillna(method='ffill')
test_data['Title'] = test_data['Name'].map(lambda x: substrings_in_string(x, title_list))
test_data['Title'] = test_data.apply(replace_titles, axis=1)

HotTitle = GetHotSex(test_data['Title'])
HotSex = GetHotSex(test_data['Sex'])
HotParch = GetHotSex(test_data['Parch'])
BinnedAge = GetBinnedAges(test_data['Age'])

X_submit = test_data[features]
X_submit.Sex = HotSex
X_submit.Age = BinnedAge
X_submit.Title = HotTitle


rf_model.fit(X, y) # use entire dataset for training

test_predictions = rf_model.predict(X_submit)

output = pandas.DataFrame({'PassengerId' : test_data.PassengerId,
                           'Survived' : test_predictions})

output.to_csv('submission.csv', index=False)



