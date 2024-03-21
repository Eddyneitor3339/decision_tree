# IMPORTAR LIBRERÍAS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# IMPORTAR BASE DE DATOS

raw_data = pd.read_csv(r'C:\Users\edd_3\OneDrive\Escritorio\Coursera\data_analysis_python\semana_2\mushrooms.csv')
#print(raw_data.columns)

data = pd.DataFrame(raw_data, columns = raw_data.columns)     # Ver cómo hacer este DataFrame más automático.
#print(data)
#print(data.info)

# VISUALIZACIÓN

sns.relplot(data = data, x = 'cap-color', y = 'cap-shape', hue = 'class')
#plt.show()

# Convert data in letter to numbers 

from sklearn.preprocessing import LabelEncoder

last = len(data.columns)
X = data[data.columns[1:last]]
y = data[data.columns[0]]

#print(y)
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

for i in X:
    X[i] = le.fit_transform(X[i])
    #print(X[i])

#print(X)

# SPLIT AND TRAIN MODEL

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training

classifer = DecisionTreeClassifier(criterion = 'gini')
model = classifer.fit(X_train, y_train )

y_predict = classifer.predict(X_test)

print(y_predict)

# EVALUATION

text_representation = tree.export_text(classifer)
print(text_representation)


fig = plt.figure(figsize=(10,8))
_ = tree.plot_tree(classifer, 
                   feature_names = data.columns, 
                   class_names = data["class"],
                   filled=True)
#plt.show()

#print(confusion_matrix(y_test, y_predict))
#print('Accuracy:', accuracy_score(y_test,y_predict))

# ENCONTRAR LOS VALORES ÓTPIMOS 

def tree_depth_tuning(d):
  classifier = DecisionTreeClassifier(max_depth=d)
  classifier.fit(X_train, y_train)

  y_pred = classifier.predict(X_test)
  accuracy = accuracy_score(y_test,y_pred)
  return accuracy

tree_results = pd.DataFrame({'D':np.arange(1, 10)})
tree_results['Accuracy'] = tree_results['D'].apply(tree_depth_tuning)
print(tree_results)


# Evaluando los 3 tipos de criterion con sus respectivos Depth

# Create a list of splitting criteria
splitting_criteria = ['entropy', 'gini', 'log_loss']

# Create a list of max depth values
max_depth_values = range(1, 20, 2)

# Create an empty dictionary to store the results
results = {}

# Train Decision Tree models with different splitting criteria and max depth
for criterion in splitting_criteria:
    for max_depth in max_depth_values:
        dt_model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
        dt_model.fit(X_train, y_train)
        y_pred = dt_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[(criterion, max_depth)] = {
            'model': dt_model,
            'accuracy': accuracy
        }

print("Results of Decision Tree Classification:")
for (criterion, max_depth), result in results.items():
    print(f"Criterion: {criterion}, Max Depth: {max_depth}, Accuracy: {result['accuracy']:.2f}")
