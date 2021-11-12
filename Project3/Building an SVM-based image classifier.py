from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

face_data = fetch_lfw_people(min_faces_per_person=80)
X = face_data.data
Y = face_data.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)

clf = SVC(class_weight='balanced', random_state=42)

parameters = {'C': [0.1, 1, 10],
              'gamma': [1e-07, 1e-08, 1e-06],
              'kernel': ['rbf', 'linear']}

grid_search = GridSearchCV(clf, parameters, n_jobs=-1, cv=5)
grid_search.fit(X_train, Y_train)
print('The best model:\n', grid_search.best_params_)
print('The best averaged performance:', grid_search.best_score_)

clf_best = grid_search.best_estimator_
pred = clf_best.predict(X_test)
print(f'The accuracy is: {clf_best.score(X_test, Y_test)*100:.1f}%')
print(classification_report(Y_test, pred, target_names=face_data.target_names))