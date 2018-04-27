# coding: utf-8

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import Normalizer  # , StandardScaler
from sklearn.svm import SVC
from sklearn import decomposition
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Importuję dane:
dane = pd.read_csv('dane.csv', sep=",")
df = pd.DataFrame(dane)
print(df.head(6))


# Rozdzielam dane na zbiór uczący i testowy:
X_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
y_column = 'species'
X = df[X_columns]
y = df[y_column]
X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.15)


sns.pairplot(df, hue="species")
plt.show()


# Z wykresów (na głównej osi) możemy zauważyć, 
# że setosa charakteryzuje się bardzo niskimi wartościami petal_width, petal_length. Co więcej wartości
# sepal_length również są małe, choć nie odbiegają już tak drastycznie od pozostałych.
# 
# Versicolor ma tendencje do posiadania wartości raczej, nie charakteryzujących się ani wysokimi,
# ani niskimi wartościami.
# 
# Virginica posiada wartości podobne do Versicolor (a czasem trochę wyższe).
# 
# Z pozostałych wykresów można zaobserwować, 
# że petal_length i petal_width są prawdopodobnie mocno skorelowane ze sobą 
# (podobnie petal_length i sepal_length, choć setosa trochę wyłamuje się z tej zależności).


# Normalizuję dane
scaler = Normalizer()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)


# C_range = 10.0 ** np.arange(-4, 4)
gamma_range = 10.0 ** np.arange(-6, 6)

C_range = 0.1 * np.arange(1, 20)
n_components = [1]  # [1,2,3]

param_grid = dict(clf__gamma=gamma_range.tolist(), 
                  clf__C=C_range.tolist(), 
                  pca__n_components=n_components)

pca = decomposition.PCA()

clf = SVC(kernel='rbf', gamma=0.1)
pipe = Pipeline(steps=[('pca', pca), ('clf', clf)])

pca.fit(X_train)
plt.figure(1, figsize=(4, 3))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')

svm = GridSearchCV(pipe, param_grid, cv=4)
svm.fit(X_train_std, y_train)


plt.axvline(svm.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label='n_components chosen')
plt.legend(prop=dict(size=12))
plt.show()


# Wyliczam score i odpowiednie parametry
print('-------------------------')
print('Best cross-validation SVC score:')
print(svm.best_score_)
print('-------------------------')
print('Number of PCA-components:')
print(svm.best_estimator_.named_steps['pca'].n_components)
print('-------------------------')
print('SVC C-parameter:')
print(svm.best_estimator_.named_steps['clf'].C)
print('-------------------------')
print('SVC Gamma-parameter:')
print(svm.best_estimator_.named_steps['clf'].gamma)
print('-------------------------')
print('Final test score')
score_SVM = svm.score(X_test_std, y_test)
print(score_SVM)
print('-------------------------')


# Klasyfikator dyskryminuje dobrze przy użyciu powyższych parametrów. 
# Użyłem PCA w celu sprawdzenia, jak model będzie zachowywać się przy użyciu innej ilości 
# zmiennych składowych - przy użyciu przy użyciu standaryzacji, dla 1 lub 2 składowych, 
# jakość modelu znacząco spadała (wówczas score wynosił 0.78).
# 
# Zmiana ze StandardScaler na Normalizer (ze standaryzacji na normalizację) sprawiła, 
# że jakość modelu wzrosła do 0.95, nawet dla 1/2 składowych.
# Ponieważ dane są bardzo wyraźnie dyskryminowane przez cechy podejrzewam,
# że nie powinno być problemu z overfittingiem, pomimo znacznej korelacji niektórych z cech 
# (dlatego użyłem PCA do ograniczenia tego problemu) 
# (choć oczywiście dobrze byłoby to sprawdzić na innych/dodatkowych danych).
# 
# Za pomocą Grid Search-u, dobrałem odpowiednie Gamma i C. Użyłem Support Vector Classifier-u 
# (ewentualnie, można spróbować za pomocą Random Forests, choć SVC dał dobre wyniki).
# 
# Na koniec spróbujemy zobaczyć, jak bardzo score różni się od przypadku, gdy użyjemy Random Forests,
# z parametrem max_features równym 1.


rf = RandomForestClassifier(n_estimators=10, max_features=1, max_depth=2)
rf.fit(X_train_std, y_train)


print('Final SVC test score, with PCA (for 1 component)')
print(score_SVM)
print('-------------------------')
print('Final Random forest test score (max_features=1)')
score_randomforests = rf.score(X_test_std, y_test)
print(score_randomforests)
print('-------------------------')


# Obserwacja numer 83 została błędnie przyporządkowana jako virginica, jednakże jej cechy są mocno zbliżone bardziej do obserwacji z gatunku virginica, niż versicolor.
prediction = svm.predict(X_test_std)
X_test.loc[:,'y_true'] = y_test
X_test.loc[:,'y_output'] = prediction
print(X_test.head(20))
#df_test.to_csv('output.csv')