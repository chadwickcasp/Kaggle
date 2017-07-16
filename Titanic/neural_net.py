import numpy as np
# import tensorflow as tf
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam
from keras.utils import plot_model
import matplotlib.pyplot as plt
import re
import tensorflowvisu as tfvu


df = pd.read_csv('train.csv')
training_data = df.copy()
labels = training_data['Survived']
test = pd.read_csv('test.csv')
# print(df['PassengerId'])
datavis = tfvu.MnistDataVis

def title_extractor(string):
    match_object = re.search(', (.+?\.) .', string)
    title_string = match_object.group(1)
    return title_string

def cabin_num_extractor(cabin_string):
    split = cabin_string.split(' ')
    if len(split) > 1:
        num_list = []
        for cabin in split:
            if cabin[1:]:    
                num_list.append(int(cabin[1:]))
        number = int(pd.DataFrame(num_list).median())
    else:
        if not split[1:]:
            return 
        number = int(split[1:])
    return number

names = training_data['Name']
titles = pd.DataFrame()
titles['Title'] = training_data['Name'].map(title_extractor)
titles_count = pd.DataFrame()
titles_count['Counts'] = titles['Title'].value_counts()
title_categories = titles_count.index.tolist()
# plt.figure(0)
# titles_count.plot(kind = 'bar')
# plt.axhline(0, color='k')

Title_Dict = {
                'Mr.':          'Normal',
                'Miss.':        'Normal',
                'Mrs.':         'Normal',
                'Master.':      'Important',
                'Dr.':          'Important',
                'Rev.':         'Important',
                'Mlle.':        'Normal',
                'Major.':       'Important',
                'Col.':         'Important',
                'Capt.':        'Important',
                'Sir.':         'Nobility',
                'Jonkheer.':    'Nobility',
                'Don.':         'Nobility',
                'the Countess.':'Nobility',
                'Ms.':          'Normal',
                'Mme.':         'Normal',
                'Lady.':        'Nobility'
             }
titles['Title'] = titles['Title'].map(Title_Dict)
titles['Title'] = pd.get_dummies(titles['Title'])
# print(titles['Title'])

sexes = pd.DataFrame()
sexes['Genders'] = training_data['Sex']
sexes['Genders'] = pd.get_dummies(sexes['Genders'])

male_age = pd.DataFrame()
female_age = pd.DataFrame()
average_age_male = training_data[training_data['Sex'] == 'male']['Age'].mean()
average_age_female = training_data[training_data['Sex'] == 'female']['Age'].mean()
male_age['Age'] = training_data[training_data['Sex'] == 'male']['Age'].fillna(average_age_male)
female_age['Age'] = training_data[training_data['Sex'] == 'female']['Age'].fillna(average_age_female)
ages = male_age.append(female_age, ignore_index=True)

training_data['Cabin'] = training_data['Cabin'].fillna('U')
cabin_letters = pd.DataFrame()
cabin_letters['Cabin Letters'] = training_data['Cabin'].map(lambda c: c[0])
cabin_letters['Cabin Letters'] = pd.get_dummies(cabin_letters['Cabin Letters'])

cabin_numbers = pd.DataFrame()
cabin_numbers['Cabin Numbers'] = training_data['Cabin'].map(cabin_num_extractor)
avg_cabin_number = cabin_numbers['Cabin Numbers'].mean()
cabin_numbers['Cabin Numbers'] = cabin_numbers['Cabin Numbers'].fillna(avg_cabin_number)

embarked_locs = pd.DataFrame()
embarked_locs['Embarked'] = training_data['Embarked']
embarked_locs['Embarked'] = pd.get_dummies(embarked_locs['Embarked'])

pclass = pd.DataFrame()
pclass['Pclass'] = training_data['Pclass']

training_data.drop(training_data.columns[[0,1]],
                   axis=1, inplace=True)
df_list  = [titles['Title'], pclass['Pclass'],
            sexes['Genders'], ages['Age'], cabin_letters['Cabin Letters'],
            cabin_numbers['Cabin Numbers'], embarked_locs['Embarked']]
train = pd.DataFrame()
train.append(df_list, ignore_index=True)
# df_labels = [df.columns.values for df in df_list]
# print(df_labels)
# df_labels = reduce(lambda x,y: x + y, df_labels)
# print(df_labels)
train2 = pd.concat(df_list, axis=1)

print(train2)
# print(training_data['Pclass'])
# print(train2['Pclass'])

model = Sequential()
model.add(Dense(train2.shape[1], input_shape=train2.shape[1:]))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

sgd = SGD(lr=3e-4, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

history = model.fit(train2.as_matrix(), labels.as_matrix(),
          validation_split=0.2,
          epochs=50)
print(history.history.keys())
fig = plt.figure(1)
ax = fig.add_subplot(1,1,1)
ax.plot(history.history['acc'], label='Accuracy')
ax.plot(history.history['val_acc'], label='Val. Accuracy')
ax.legend(loc='upper right', shadow=True)
# plt.show()

fig = plt.figure(2)
ax = fig.add_subplot(1,1,1)
ax.plot(history.history['loss'], label='Loss')
ax.plot(history.history['val_loss'], label='Val. Loss')
ax.legend(loc='upper right', shadow=True)
plt.show()

plot_model(model, to_file='model.png')

score = model.evaluate()
