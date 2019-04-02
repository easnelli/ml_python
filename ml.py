# coding=utf-8
from __future__ import division
from sklearn import datasets
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split as tts

# svm: significant accuracy with less computation power. Kan brukes til både regressjon og klassifisering (mest her).
# tree: decision trees. trestruktur (som binærtrær).
# vanlig å bruke 90% til training og 10% til testing
wine = datasets.load_wine()

features = wine.data
labels = wine.target



train_feats, test_feats, train_labels, test_labels = tts(features, labels, test_size = 0.2)

#SVC = support vector characterisation
clf = svm.SVC()

# train with svm
# Det vi sier: se på disse labelsene og disse featuresene og finn en sammenheng mellom de.
clf.fit(train_feats, train_labels)

#predictions
predictions = clf.predict(test_feats)
print predictions

score = 0
for i in range(len(predictions)):
    if predictions[i] == test_labels[i]:
        score += 1
print "Accuracy with svm: ", score / len(predictions)

# train with tree

clf_2 = tree.DecisionTreeClassifier()

# train
# Det vi sier: se på disse labelsene og disse featuresene og finn en sammenheng mellom de.
clf_2.fit(train_feats, train_labels)

#predictions
predictionsWithTree = clf_2.predict(test_feats)
print predictionsWithTree

scoreWithTree = 0
for i in range(len(predictionsWithTree)):
    if predictionsWithTree[i] == test_labels[i]:
        scoreWithTree += 1


print "Accuracy with tree: ", scoreWithTree / len(predictionsWithTree)

# train with RandomForestClassifier

clf_3 = RandomForestClassifier()

# train
# Det vi sier: se på disse labelsene og disse featuresene og finn en sammenheng mellom de.
clf_3.fit(train_feats, train_labels)

#predictions
predictionsWithRandom = clf_3.predict(test_feats)
print predictionsWithRandom

scoreWithRandom = 0
for i in range(len(predictionsWithRandom)):
    if predictionsWithRandom[i] == test_labels[i]:
        scoreWithRandom+= 1


print "Accuracy with RandomForestClassifier: ", scoreWithRandom / len(predictionsWithRandom)
