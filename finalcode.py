"""
Alex Todorovic
CMPS-3240
Final Code

Using pandas for data processing,  GradientBoostingClassifier from sklearn.ensemble
for training and testing, metrics from sklearn for evaluating accuracy.

This learning model is purposed towards teaching computers to understand soccer match
statistics. By training on game data with features such as shots, corners, etc. the
model assigns weights to features that influence a game. For example, if team A is having
significantly more shots on goal than team B, then team A is more likely to win.
The goal is for the model to have the ability of "watching" a game and understanding,
based on statistics, that one team may be outperforming another, or that a game is very
close to call, etc.
"""
from sklearn import *
import pandas as pd
from sklearn.ensemble import  GradientBoostingClassifier



'''Training data preprocessing'''
league = pd.read_csv('E0-3.csv', header=None,
                                                
                         names=['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR',
                                'Referee', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR',
                                'bet1', 'bet2', 'bet3', 'bet4', 'bet5', 'bet6', 'bet7', 'bet8', 'bet9', 'bet10', 'bet11',
                                'bet12', 'bet13', 'bet14', 'bet15', 'bet16', 'bet17', 'bet18', 'bet19', 'bet20', 'bet21',
                                'bet22', 'bet23','bet24', 'bet25', 'bet26', 'bet27', 'bet28', 'bet29', 'bet30', 'bet31',
                                'bet32','bet33', 'bet34', 'bet35', 'bet36', 'bet37', 'bet38', 'bet39', 'bet40', 'bet41',
                                'bet42'])


'''Features list: home shots, away shots, home fouls, away fouls,
home corners, away corners, home yellow cards, away yellow
cards, home red cards, away red cards'''

features = ['HS', 'AS', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR',]
x_train = league[features]
x_train = x_train[1:] #removing labels
y_train = league[['FTR']]
y_train = y_train[1:] #removing labels


'''     Testing data preprocessing     '''
league = pd.read_csv('E0-2.csv', header=None,

                         names=['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR',
                                'Referee', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR',
                                'bet1', 'bet2', 'bet3', 'bet4', 'bet5', 'bet6', 'bet7', 'bet8', 'bet9', 'bet10', 'bet11',
                                'bet12', 'bet13', 'bet14', 'bet15', 'bet16', 'bet17', 'bet18', 'bet19', 'bet20', 'bet21',
                                'bet22', 'bet23','bet24', 'bet25', 'bet26', 'bet27', 'bet28', 'bet29', 'bet30', 'bet31',
                                'bet32','bet33', 'bet34', 'bet35', 'bet36', 'bet37', 'bet38', 'bet39', 'bet40', 'bet41',
                                'bet42'])

'''Doing the same thing in training data processing, except I'm using a different .csv file'''
x_test = league[features]
x_test = x_test[1:] #removing labels
y_test = league[['FTR']]
y_test = y_test[1:] #removing labels




'''Evaluating accuracy, precision, recall, and f1 scores. '''
n = 50 #iteration count for averaging scores - feel free to change it
accuracy = 0
precision = 0
recall = 0
f1 = 0

booster = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

for i in range(n):
    booster.fit(x_train, y_train)
    y_pred = booster.predict(x_test)

    accuracy += metrics.accuracy_score(y_test, y_pred)                      #accuracy
    precision += metrics.precision_score(y_test, y_pred, average='macro')   #precision
    recall += metrics.recall_score(y_test, y_pred, average='macro')         #recall
    f1 += metrics.f1_score(y_test, y_pred, average='macro')                 #f1

print('~~~Evaluating Gradient Boosting Classifier~~~')
print('Accuracy score: ' + str(accuracy / n) + '\n')
print('Precision score: ' + str(precision / n) + '\n')
print('Recall score: ' + str(recall / n) + '\n')
print('F1 score: ' + str(f1 / n) + '\n')
