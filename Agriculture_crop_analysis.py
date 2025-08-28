import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
df = pd.read_excel("C:\\Users\\HP\\OneDrive\\Desktop\\Sample_Crop_Data.xlsx")
print(df.head())

X = df[['Nitrogen','Phosphorous','Potassium','pH_value']]
y = df['Corresponding_crop_label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['Nitrogen','Phosphorous','Potassium','pH_value']],df['Corresponding_crop_label'], test_size=0.2)

lr = LogisticRegression(max_iter=200)
print(lr.fit(X_train,y_train))
print(f"Accuracy for Logistic Regression: {lr.score(X_test,y_test)}")
svm = SVC()
print(svm.fit(X_train,y_train))
print(f"Accuracy for SVC: {svm.score(X_test,y_test)}")
rfc = RandomForestClassifier(n_estimators=400, max_features='sqrt', max_depth=None)
print(rfc.fit(X_train,y_train))
print(f"Accuracy for RandomForestClassifier: {rfc.score(X_test,y_test)}")

def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train,y_train)
    return model.score(X_test,y_test)

print(get_score(LogisticRegression(max_iter=200), X_train, X_test, y_train, y_test))
print(get_score(SVC(), X_train, X_test, y_train, y_test))
print(get_score(RandomForestClassifier(), X_train, X_test, y_train, y_test))

from sklearn.model_selection import StratifiedKFold
stkf = StratifiedKFold(n_splits=5)
from sklearn.model_selection import cross_val_score

print(cross_val_score(LogisticRegression(max_iter=200),X,y, cv=stkf))
print(cross_val_score(SVC(),X,y, cv=stkf))
print(cross_val_score(RandomForestClassifier(),X,y, cv=stkf))

print("\n")
print(rfc.predict(scaler.transform([[124.1920994, 113.6995156, 111.5964029, 4.512839813]])))
print(rfc.predict(scaler.transform([[52.435617, 31.844275, 56.032608, 7.363515]])))
print("\n")

from sklearn.metrics import confusion_matrix
#since RandomForestClassifier shows the best result
y_predicted = rfc.predict(X_test)
cm = confusion_matrix(y_test,y_predicted)
print(cm)
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

#shape of the DataFrame for crop label 1 and 0
print("Shape of crop label 1:", df[df['Corresponding_crop_label'] == 1].shape)
print("Shape of crop label 0:", df[df['Corresponding_crop_label'] == 0].shape)

#data types of the Soil Features
print("Data Types:\n", df.dtypes)

# Calculate the mean for numeric columns grouped by 'Corresponding_crop_label'
mean_values = df.groupby('Corresponding_crop_label').mean(numeric_only=True)

# Print mean values
print("Mean Values:\n", mean_values)

# Plotting the mean values
mean_values.plot(kind='bar', figsize=(10, 6))
plt.xlabel('Crop Type')
plt.ylabel('Soil Features')
plt.xticks(rotation=0)
plt.legend(title='Factors')
plt.tight_layout()
plt.show()

from sklearn.neural_network import MLPClassifier
model = MLPClassifier(hidden_layer_sizes=60)
print(model.fit(X_train,y_train))
print(f"train accuracy {model.score(X_train,y_train)}")
print(f"test accuracy {model.score(X_test,y_test)}")