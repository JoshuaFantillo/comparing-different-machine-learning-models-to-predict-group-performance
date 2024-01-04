################################################################
# Author: Joshua Fantillo
# File: neural_network_models.py
# Date: April 11, 2022
# Employer: The University of the Fraser Valley
# Location: Abbotsford, BC, Canada
# Description: This file builds and trains the machine learning 
# models. It puts them into a dataframe for comparison. 
################################################################

from sklearn import svm
import csv
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# gets the dataframe from the csv file
def get_data_frame():
	return pd.read_csv('research_data')

# runs the file
def main():
	data_frame = get_data_frame()
	get_all_models(data_frame)

# This is used to get many different run throughs of the machine learning models
def get_all_models(data_frame):
	mlp_prec = 0
	mlp_rec = 0 
	mlp_fs = 0
	mlp_acc = 0
	knn_prec = 0
	knn_rec = 0 
	knn_fs = 0
	knn_acc = 0
	rf_prec = 0
	rf_rec = 0 
	rf_fs = 0
	rf_acc = 0
	gnb_prec = 0
	gnb_rec = 0
	gnb_fs = 0 
	gnb_acc = 0
	mnb_prec = 0 
	mnb_rec = 0 
	mnb_fs = 0 
	mnb_acc = 0
	svm_prec = 0 
	svm_rec = 0 
	svm_fs = 0
	svm_acc = 0
	
	mlp_me = 0
	mlp_mse = 0
	mlp_r2 = 0
	log_me = 0
	log_mse = 0
	log_r2 = 0
	rf_me = 0 
	rf_mse = 0
	rf_r2 = 0
	svm_me = 0
	svm_mse = 0
	svm_r2 = 0
	
	# this gets 28 different training and testing sets for the models
	for i in range(28):
		X_train, X_test, y_train, y_test = get_train_and_testing_datasets(data_frame, i)
		X_train_scaled, X_test_scaled = scale(X_train, X_test)
		y_train_grade = get_grade(y_train)
		y_test_grade = get_grade(y_test)
		hold_prec, hold_rec, hold_fs, hold_acc = get_MLP_classifier(X_train_scaled, X_test_scaled, y_train_grade, y_test_grade)
		mlp_prec += hold_prec
		mlp_rec += hold_rec
		mlp_fs += hold_fs
		mlp_acc += hold_acc
		
		hold_me, hold_mse, hold_r2 = get_MLP_regressor(X_train_scaled, X_test_scaled, y_train, y_test)
		mlp_me += hold_me
		mlp_mse += hold_mse
		mlp_r2 += hold_r2
		
		hold_prec, hold_rec, hold_fs, hold_acc = get_knn(X_train_scaled, X_test_scaled, y_train_grade, y_test_grade)
		knn_prec += hold_prec
		knn_rec += hold_rec
		knn_fs += hold_fs
		knn_acc += hold_acc
		
		hold_me, hold_mse, hold_r2 = get_logistic_regression(X_train_scaled, X_test_scaled, y_train, y_test)
		log_me += hold_me
		log_mse += hold_mse
		log_r2 += hold_r2
		
		hold_prec, hold_rec, hold_fs, hold_acc = get_random_forest_clf(X_train_scaled, X_test_scaled, y_train_grade, y_test_grade)
		rf_prec += hold_prec
		rf_rec += hold_rec
		rf_fs += hold_fs
		rf_acc += hold_acc
		
		hold_me, hold_mse, hold_r2 = get_random_forest_reg(X_train_scaled, X_test_scaled, y_train, y_test)
		rf_me += hold_me
		rf_mse += hold_mse
		rf_r2 += hold_r2
		
		hold_prec, hold_rec, hold_fs, hold_acc = get_gnb(X_train_scaled, X_test_scaled, y_train_grade, y_test_grade)
		gnb_prec += hold_prec
		gnb_rec += hold_rec
		gnb_fs += hold_fs
		gnb_acc += hold_acc
		
		hold_prec, hold_rec, hold_fs, hold_acc = get_mnb(X_train, X_test, y_train_grade, y_test_grade)
		mnb_prec += hold_prec
		mnb_rec += hold_rec
		mnb_fs += hold_fs
		mnb_acc += hold_acc
		
		hold_prec, hold_rec, hold_fs, hold_acc = get_svmc(X_train_scaled, X_test_scaled, y_train_grade, y_test_grade)
		svm_prec += hold_prec
		svm_rec += hold_rec
		svm_fs += hold_fs
		svm_acc += hold_acc
		
		hold_me, hold_mse, hold_r2 = get_svmr(X_train_scaled, X_test_scaled, y_train, y_test)
		svm_me += hold_me
		svm_mse += hold_mse
		svm_r2 += hold_r2
	
	# saves it to two different dataframes so its easier to view.
	# one dataframe for classifiers and one for regressors	
	clf_df = pd.DataFrame()
	reg_df = pd.DataFrame()
	
	prec_col = get_clf_col(mlp_prec, knn_prec, rf_prec, gnb_prec, mnb_prec, svm_prec)
	rec_col = get_clf_col(mlp_rec, knn_rec, rf_rec, gnb_rec, mnb_rec, svm_rec)
	fs_col = get_clf_col(mlp_fs, knn_fs, rf_fs, gnb_fs, mnb_fs, svm_fs)
	acc_col = get_clf_col(mlp_acc, knn_acc, rf_acc, gnb_acc, mnb_acc, svm_acc)
	
	me_col = get_reg_col(mlp_me, log_me, rf_me, svm_me)
	mse_col = get_reg_col(mlp_mse, log_mse, rf_mse, svm_mse)
	r2_col = get_reg_col(mlp_r2, log_r2, rf_r2, svm_r2)
	
	clf_df['Precision Scores'] = prec_col
	clf_df['Recall Scores'] = rec_col
	clf_df['F1 Scores'] = fs_col
	clf_df['Accuracy Scores'] = acc_col
	
	clf_df.index = ["MLP CLF", 'KNN', 'Random Forest CLF', 'GNB', 'MNB', 'SVM CLF']
	
	reg_df['Mean Error'] = me_col
	reg_df['Mean Squared Error'] = mse_col
	reg_df['R2 Score'] = r2_col
	
	reg_df.index = ['MLP Reg', 'Logistic Reg', 'Random Forest Reg', 'SVM Reg']
	
	print(clf_df)
	print(reg_df)
	
	
# gets the average score of each regressor value (ME, MSE, R2)
def get_reg_col(first, second, third, fourth):
	column = []
	column.append(first/28)
	column.append(second/28)
	column.append(third/28)
	column.append(fourth/28)
	return column
	
	
# gets the average score of each classifier value (Precision, Recall, F1 Score, Accuracy)
def get_clf_col(first, second, third, fourth, fifth, sixth):
	column = []
	column.append(first/28)
	column.append(second/28)
	column.append(third/28)
	column.append(fourth/28)
	column.append(fifth/28)
	column.append(sixth/28)
	return column

# gets the training and testing dataframe
# depending on what i is the training and testing data is different over each iteration 
def get_train_and_testing_datasets(data_frame, i):
	X = data_frame.drop(['AGS'], axis=1)
	y = data_frame['AGS']
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=i, test_size=0.2)
	return X_train, X_test, y_train, y_test

# converts the percent to a letter grade
# (0-50 : F, 50-60 : D, 60-73 : C, 73-86 : B, 86-100 : A)
def get_grade(y_data):
	data_frame = pd.DataFrame()
	y_grade = []
	for item in y_data:
		if item < 50:
			y_grade.append(0)
		if item >= 50 and item < 60:
			y_grade.append(1)
		if item >= 60 and item < 73:
			y_grade.append(2)
		if item >= 73 and item < 86:
			y_grade.append(3)
		if item >= 86:
			y_grade.append(4)
	
	data_frame['GPA'] = y_grade

	return data_frame

# scales the dataframe	
def scale(train, test):
	sc = StandardScaler()
	sc.fit(train)
	trainscaled=sc.transform(train)
	testscaled=sc.transform(test)
	return trainscaled, testscaled

# trains the support vm classifier
def get_svmc(X_train, X_test, y_train, y_test):
	support_vm = svm.SVC().fit(X_train, np.ravel(y_train))
	y_pred = support_vm.predict(X_test)
	prec, rec, fs, acc = get_prec_recall_fscore(y_test,y_pred)
	return prec, rec, fs, acc

# gets teh support vm regressor
def get_svmr(X_train, X_test, y_train, y_test):
	support_vm = svm.SVR().fit(X_train, np.ravel(y_train))
	y_pred = support_vm.predict(X_test)
	me, mse = get_mean_squared_error(y_test, y_pred)
	r2 = get_r2_score(y_test, y_pred)
	return me, mse, r2

# trains the guassian naive basis model
def get_gnb(X_train, X_test, y_train, y_test):
	gnb = GaussianNB().fit(X_train, np.ravel(y_train))
	y_pred = gnb.predict(X_test)
	prec, rec, fs, acc = get_prec_recall_fscore(y_test,y_pred)
	return prec, rec, fs, acc

# trainsthe multinomial naive basis model
def get_mnb(X_train, X_test, y_train, y_test):
	mnb = MultinomialNB().fit(X_train, np.ravel(y_train))
	y_pred = mnb.predict(X_test)
	prec, rec, fs, acc = get_prec_recall_fscore(y_test,y_pred)
	return prec, rec, fs, acc

# trains the MLP Classifier
def get_MLP_classifier(X_train, X_test, y_train, y_test):
	clf = MLPClassifier(alpha=0.009,hidden_layer_sizes=(16,16,16,16),activation="relu", random_state=1,max_iter=2000).fit(X_train, np.ravel(y_train))
	y_pred = clf.predict(X_test)
	prec, rec, fs, acc = get_prec_recall_fscore(y_test,y_pred)
	return prec, rec, fs, acc

# trains the MLP Regressor
def get_MLP_regressor(X_train, X_test, y_train, y_test):
	reg = MLPRegressor(alpha=0.1, hidden_layer_sizes=(32,32,32), activation="relu", random_state=1, max_iter=2000).fit(X_train, np.ravel(y_train))
	y_pred=reg.predict(X_test)
	me, mse = get_mean_squared_error(y_test, y_pred)
	r2 = get_r2_score(y_test, y_pred)
	return me, mse, r2

# trains knn model
def get_knn(X_train, X_test, y_train, y_test):
	knn = KNeighborsClassifier(n_neighbors=1)
	knn.fit(X_train, np.ravel(y_train))
	y_pred = knn.predict(X_test)
	prec, rec, fs, acc = get_prec_recall_fscore(y_test,y_pred)
	return prec, rec, fs, acc

# trains logistic regression model	
def get_logistic_regression(X_train, X_test, y_train, y_test):
	reg = LogisticRegression(random_state=0)
	reg.fit(X_train, np.ravel(y_train))
	y_pred = reg.predict(X_test)
	me, mse = get_mean_squared_error(y_test, y_pred)
	r2 = get_r2_score(y_test, y_pred)
	return me, mse, r2

# trains the randon forest regression model
def get_random_forest_reg(X_train, X_test, y_train, y_test):
	reg = RandomForestRegressor(max_depth=2, random_state=0)
	reg.fit(X_train, np.ravel(y_train))
	y_pred = reg.predict(X_test)
	me, mse = get_mean_squared_error(y_test, y_pred)
	r2 = get_r2_score(y_test, y_pred)
	return me, mse, r2
	
# trains the random forest clf model
def get_random_forest_clf(X_train, X_test, y_train, y_test):
	clf = RandomForestClassifier(max_depth=2, random_state=0)
	clf.fit(X_train, np.ravel(y_train))
	y_pred = clf.predict(X_test)
	prec, rec, fs, acc = get_prec_recall_fscore(y_test,y_pred)
	return prec, rec, fs, acc

# gets the mean error and mean square error
def get_mean_squared_error(y_test, y_pred):
	me = mean_squared_error(y_test, y_pred, squared=False)
	mse = mean_squared_error(y_test, y_pred)
	return me, mse

# gets the r2 score	
def get_r2_score(y_test, y_pred):
	r2 = r2_score(y_test, y_pred)
	return r2

#gets the precision, recall, fscore, and accuracy
def get_prec_recall_fscore(y_test,y_pred):
	y_test = np.array(y_test)
	y_pred = np.array(y_pred)
	prec = precision_score(y_test, y_pred, average='weighted')
	rec = recall_score(y_test, y_pred, average='weighted')
	fs = f1_score(y_test, y_pred, average='weighted')
	acc = accuracy_score(y_test, y_pred)
	return prec, rec, fs, acc
	
if __name__ == "__main__":
	main()
	
