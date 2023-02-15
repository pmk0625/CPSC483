import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse, r2_score
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.model_selection import cross_val_score, RepeatedKFold

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("Data1.csv")
scaled_data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
scaled_data = pd.DataFrame(scaled_data, columns=['T', 'P', 'TC', 'SV', 'Idx'])
X, y = data.iloc[:, 0:-1], data.iloc[:, -1]
scaled_X, scaled_y = scaled_data.iloc[:, 0:-1], scaled_data.iloc[:, -1]

print("===[Least Square]===")
print("{:<10} {:<10} {:<10}".format("Degree", "RMSE", "R^2"))

# LinearRegression Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

lr_model = LinearRegression().fit(X_train, y_train)
y_preds = lr_model.predict(X_test)

print("{:<10} {:<10} {:<10}".format("1", str(round(np.sqrt(mse(y_preds, y_test)), 6)), str(round(r2_score(y_preds, y_test), 6))))

# NonLinearRegression Model
for n in range(2, 20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)
    
    poly_func = PolynomialFeatures(degree = n)
    X_train, X_test = poly_func.fit_transform(X_train), poly_func.fit_transform(X_test)
    
    nlr_model = LinearRegression().fit(X_train, y_train)
    y_preds = nlr_model.predict(X_test)

    print("{:<10} {:<10} {:<10}".format(str(n), str(round(np.sqrt(mse(y_preds, y_test)), 6)), str(round(r2_score(y_preds, y_test), 6))))


print("===[Gradient Descent]===")
print("{:<10} {:<10} {:<10}".format("Degree", "RMSE", "R^2"))

def gradient_descent(X, y, alpha, epoch):
    X = (X - X.mean()) / X.std() # scale data

    m = X.shape[0] # size of features
    X = np.hstack((np.ones((m, 1)), X)) # add columns of 1 before X
    cost = np.zeros(epoch) # initialize cost
    n = X.shape[1]
    w = np.zeros(n) # initialize weight
    
    for i in range(epoch):
        yhat = np.dot(X, w.T) - y # do prediction
        cost[i] = 1 / (2 * m) * np.dot(yhat.T, yhat) # calculate cost
        w = w - (alpha * (1 / m) * np.dot(X.T, yhat)) # update weight
    
    return cost, w

# LinearRegression Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)
cost, w = gradient_descent(X_train, y_train, 0.1, 5000)

X_test = (X_test - X_test.mean()) / X_test.std()
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
y_preds = np.dot(X_test, w.T)

print("{:<10} {:<10} {:<10}".format("1", str(round(np.sqrt(mse(y_preds, y_test)), 6)), str(round(r2_score(y_preds, y_test), 6))))

# NonLinearRegression Model
from sklearn.preprocessing import PolynomialFeatures

for n in range(2, 3):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)
    
    poly_func = PolynomialFeatures(degree = n)
    X_train, X_test = poly_func.fit_transform(X_train), poly_func.fit_transform(X_test)
    
    cost, w = gradient_descent(X_train, y_train, 0.1, 15000)
    
    X_test = (X_test - X_test.mean()) / X_test.std()
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    y_preds = np.dot(X_test, w.T)

    print("{:<10} {:<10} {:<10}".format(str(n), str(round(np.sqrt(mse(y_preds, y_test)), 6)), str(round(r2_score(y_preds, y_test), 6))))


print("===[Least Square w/ Scaled Data]===")
print("{:<10} {:<10} {:<10}".format("Degree", "RMSE", "R^2"))

# LinearRegression Model
X_train, X_test, y_train, y_test = train_test_split(scaled_X, scaled_y, test_size = 0.30, random_state = 0)

lr_model = LinearRegression().fit(X_train, y_train)
y_preds = lr_model.predict(X_test)

print("{:<10} {:<10} {:<10}".format("1", str(round(np.sqrt(mse(y_preds, y_test)), 6)), str(round(r2_score(y_preds, y_test), 6))))

# NonLinearRegression Model
for n in range(2, 20):
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, scaled_y, test_size = 0.30, random_state = 0)
    
    poly_func = PolynomialFeatures(degree = n)
    X_train, X_test = poly_func.fit_transform(X_train), poly_func.fit_transform(X_test)
    
    nlr_model = LinearRegression().fit(X_train, y_train)
    y_preds = nlr_model.predict(X_test)

    print("{:<10} {:<10} {:<10}".format(str(n), str(round(np.sqrt(mse(y_preds, y_test)), 6)), str(round(r2_score(y_preds, y_test), 6))))

print("===[Lasso]===")
print("{:<10} {:<10} {:<10} {:<10}".format("Degree", "Score", "RMSE", "R^2"))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

lasso = Lasso(alpha = 0.1)
lasso.fit(X_train, y_train)
y_preds = lasso.predict(X_test)
print("{:<10} {:<10} {:<10} {:<10}".format("1", str(round(lasso.score(X_test, y_test), 6)), str(round(np.sqrt(mse(y_preds, y_test)), 6)), str(round(r2_score(y_preds, y_test), 6))))

for n in range(2, 10):
    lasso_poly = PolynomialFeatures(degree = n, include_bias=True)
    x_train_lasso= lasso_poly.fit_transform(X_train)
    x_test_lasso = lasso_poly.transform(X_test)

    lasso = Lasso(alpha = 0.01, tol = 1.9e-11)
    lasso.fit(x_train_lasso, y_train)
    y_preds= lasso.predict(x_test_lasso)
    print("{:<10} {:<10} {:<10} {:<10}".format(str(n), str(round(lasso.score(x_test_lasso, y_test), 6)), str(round(np.sqrt(mse(y_preds, y_test)), 6)), str(round(r2_score(y_preds, y_test), 6))))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)
print("===[Ridge]===")
print("{:<10} {:<10} {:<10} {:<10}".format("Degree", "Score", "RMSE", "R^2"))
ridge = Ridge()
ridge.fit(X_train, y_train)
y_preds = ridge.predict(X_test)

print("{:<10} {:<10} {:<10} {:<10}".format("1", str(round(ridge.score(X_test, y_test), 6)), str(round(np.sqrt(mse(y_preds, y_test)), 6)), str(round(r2_score(y_preds, y_test), 6))))

for n in range(2, 10):
    poly_ridge = PolynomialFeatures(degree = n, include_bias=True)
    x_train_ridge = poly_ridge.fit_transform(X_train)
    x_test_ridge = poly_ridge.transform(X_test)
    ridge = Ridge(alpha = 0.01)
    ridge.fit(x_train_ridge, y_train)
    y_preds = ridge.predict(x_test_ridge)
    
    print("{:<10} {:<10} {:<10} {:<10}".format(str(n), str(round(ridge.score(x_test_ridge, y_test), 6)), str(round(np.sqrt(mse(y_preds, y_test)), 6)), str(round(r2_score(y_preds, y_test), 6))))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)
print("===[ElasticNet]===")
print("{:<10} {:<10} {:<10} {:<10}".format("Degree", "Score", "RMSE", "R^2"))

EN = ElasticNet(alpha=0.01)
EN.fit(X_train, y_train)
y_preds = EN.predict(X_test)

print("{:<10} {:<10} {:<10} {:<10}".format("1", str(round(EN.score(X_test, y_test), 6)), str(round(np.sqrt(mse(y_preds, y_test)), 6)), str(round(r2_score(y_preds, y_test), 6))))

for n in range(2, 10):
    EN_poly = PolynomialFeatures(degree = n, include_bias=True)
    x_train_trans = EN_poly.fit_transform(X_train)
    x_test_trans = EN_poly.transform(X_test)

    EN = ElasticNet(alpha = 0.01, tol = 1.9e-15)
    EN.fit(x_train_trans, y_train)
    y_preds = EN.predict(x_test_trans)
    
    print("{:<10} {:<10} {:<10} {:<10}".format(str(n), str(round(EN.score(x_test_trans, y_test), 6)), str(round(np.sqrt(mse(y_preds, y_test)), 6)), str(round(r2_score(y_preds, y_test), 6))))

print("===[K-Fold]===")
#split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

#models
lasso = Lasso(alpha = 0.1)
ridge = Ridge(alpha = 0.1)
EN = ElasticNet(alpha = 0.1)

#fit
lasso.fit(X_train, y_train)
ridge.fit(X_train, y_train)
EN.fit(X_train, y_train)

#evaluation methods
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

#cv Test calculation
score_Lasso_Test = cross_val_score(lasso, X_test, y_test, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
score_Ridge_Test = cross_val_score(ridge, X_test, y_test, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
score_EN_Test = cross_val_score(EN, X_test, y_test, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

score_Lasso_Test = np.absolute(score_Lasso_Test)
score_Ridge_Test = np.absolute(score_Ridge_Test)
score_EN_Test = np.absolute(score_EN_Test)

#cv Training
score_Lasso_Train = cross_val_score(lasso, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
score_Ridge_Train = cross_val_score(ridge, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
score_EN_Train = cross_val_score(EN, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

score_Lasso_Train = np.absolute(score_Lasso_Train)
score_Ridge_Train = np.absolute(score_Ridge_Train)
score_EN_Train = np.absolute(score_EN_Train)

#print test
print("-----------Test Values-----------")
print("Lasso Test Value: %.3f (%.3f)" % (np.mean(score_Lasso_Test), np.std(score_Lasso_Test)))
print("Ridge Test Value: %.3f (%.3f)" % (np.mean(score_Ridge_Test), np.std(score_Ridge_Test)))
print("EN Test Value: %.3f (%.3f)" % (np.mean(score_EN_Test), np.std(score_EN_Test)))

print()
print("-----------Train Values-----------")

#print train
print("Lasso Train Value: %.3f (%.3f)" % (np.mean(score_Lasso_Train), np.std(score_Lasso_Train)))
print("Ridge Train Value: %.3f (%.3f)" % (np.mean(score_Ridge_Train), np.std(score_Ridge_Train)))
print("EN Train Value: %.3f (%.3f)" % (np.mean(score_EN_Train), np.std(score_EN_Train)))