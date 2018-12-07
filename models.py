import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

class Model: 

    def __init__(self):
        pass

    def logistic_regression(self,X_train_tf,X_valid_tf,y_train_tf,y_valid_tf):    
        clf = LogisticRegression(C = 5.112113111111119,penalty='l2',warm_start=True)
        clf.fit(X_train_tf, y_train_tf)
        print('Accuracy of Logistic Regression: ', clf.score( X_valid_tf, y_valid_tf))
        scores = cross_val_score(clf,X_train_tf,y_train_tf,cv=5)
        print("Cross validation scores: ", scores)
        pickle.dump(clf,open("LogisticRegression.p","wb"))
        return clf

    def svc_(self,X_train_tf,X_valid_tf,y_train_tf,y_valid_tf):    
        svc = LinearSVC(dual=False)
        svc.fit(X_train_tf, y_train_tf)
        print ("Accuracy of LinearSVC:", svc.score(X_valid_tf, y_valid_tf))
        scores = cross_val_score(svc,X_train_tf,y_train_tf,cv=5)
        print("Cross validation scores: ", scores)
        pickle.dump(svc,open("SVM.p","wb"))
        return svc

    def random_forest(self,X_train_tf,X_valid_tf,y_train_tf,y_valid_tf):    
        rf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
        rf.fit(X_train_tf, y_train_tf)
        print ("Accuracy of RandomForest Classifier:",rf.score(X_valid_tf, y_valid_tf))
        scores = cross_val_score(rf,X_train_tf,y_train_tf,cv=5)
        print("Cross validation scores: ", scores)
        pickle.dump(rf,open("RandomForest.p","wb"))
        return rf

    def ada_boost(self,X_train_tf,X_valid_tf,y_train_tf,y_valid_tf):    
        ab = AdaBoostClassifier(n_estimators=50)
        ab.fit(X_train_tf, y_train_tf)
        print ("Accuracy of Ada_Boost:",ab.score(X_valid_tf, y_valid_tf))
        scores = cross_val_score(ab,X_train_tf,y_train_tf,cv=5)
        print("Cross validation scores: ", scores)
        pickle.dump(ab,open("Adaboost.p","wb"))
        return ab
    