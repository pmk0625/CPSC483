import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler



df = pd.read_csv('Data1.csv')

vt = VarianceThreshold(threshold=0.5)
vt.fit(df)

mask = vt.get_support()

chosen_features = list(df.loc[:, mask].columns)
removed_features = [feature for feature in df.columns if feature not in chosen_features]

print(df.isnull().sum()/len(df))
print()
df.info()
print()
print('Chosen features: {}'.format(chosen_features))
print('Removed features: {}'.format(removed_features))

#RFE
X = df.drop ('TC', axis=1)
y = df['TC']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)

lr = LinearRegression()
rfe = RFE(estimator=lr, n_features_to_select=10)
rfe.fit(X_train, y_train)

print()
print('Selected features: {}'.format(list(X.columns[rfe.support_])))
print()

print(pd.DataFrame(zip(X.columns, rfe.ranking_), columns=['Feature', 'Ranking']).sort_values('Ranking'))


#plot
plt.figure(figsize=(10, 7))

mask = np.triu(np.ones_like(df.corr(), dtype=bool))

sns.heatmap(df.corr(), annot=True, mask=mask, vmin=0.5, vmax=1)
plt.title('Correlation Coefficient of Predictors')
plt.show()



