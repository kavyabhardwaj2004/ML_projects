import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
from matplotlib.pyplot import title
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("C:\\Users\\HP\\Downloads\\HRcommasep1603576336980\\HR_comma_sep.csv")
print(df.head())

#...................normalizing data before plotting..........................#

scalar = StandardScaler().fit(df[['satisfaction_level','average_montly_hours','number_project','last_evaluation','time_spend_company']])
normalized_HR_data = scalar.transform(df[['satisfaction_level','average_montly_hours','number_project','last_evaluation','time_spend_company']])
print(normalized_HR_data)

normalized_HR_data_df = pd.DataFrame(normalized_HR_data,columns=['satisfaction_level','average_montly_hours','number_project','last_evaluation','time_spend_company'])
normalized_HR_data_df = normalized_HR_data_df.join(df[df.columns.drop(['satisfaction_level','average_montly_hours','number_project','last_evaluation','time_spend_company'])])
X = normalized_HR_data_df[normalized_HR_data_df.columns.drop('left')]
Y = normalized_HR_data_df['left']
print(X.shape)
print(Y.shape)
print(X.average_montly_hours)

#..........plotting relationship between satisfaction level and working hours of employees who have left the organization..........#

left_employees = normalized_HR_data_df[normalized_HR_data_df['left']==1]
retained = normalized_HR_data_df[normalized_HR_data_df['left']==0]
print(f"employees who have left {left_employees}")
print(f"employees who have stayed {retained}")
print(normalized_HR_data_df.dtypes)
numeric_cols = normalized_HR_data_df.select_dtypes(include=[np.number]).columns
mean_values = normalized_HR_data_df.groupby('left')[numeric_cols].mean()
print(mean_values)

#.....the below model works...but it is highly unreadable....you’re grouping by two continuous numeric variables (satisfaction_level and average_montly_hours) after scaling. Since both are floats with many unique values, the groupby creates tons of bars (almost one per row), making the bar chart unreadable.
# so either of the 2 can be used....Scatter Plot (best for two continuous variables), Hexbin / Density Plot (if many points overlap)....u can even use...BIN+BAR PLOT

# satisfy_count = left_employees.groupby(['satisfaction_level','average_montly_hours']).size().unstack(fill_value=0)
# satisfy_count.plot(kind='bar', figsize=(10,7))
# plt.xlabel("features")
# plt.ylabel("average_montly_hours")
# plt.tight_layout()
# plt.legend(title='Satisfaction and average monthly hours', labels = ['stayed','left'])
# plt.show()

#...........................BIN + BAR PLOT................................#

bins = [0,0.2,0.4,0.6,0.8,1.0]
left_employees['satisfaction_bin'] = pd.cut(left_employees['satisfaction_level'], bins)
satisfy_count = left_employees.groupby('satisfaction_bin')['average_montly_hours'].mean()

satisfy_count.plot(kind='bar', figsize=(8,6))
plt.xlabel("Satisfaction Level (binned)")
plt.ylabel("Average Monthly Hours (mean)")
plt.title("Avg Monthly Hours vs Satisfaction (Employees who Left)")
plt.show()

#...........................Scatter Plot................................#

plt.figure(figsize=(10,7))
plt.scatter(
    x=left_employees['average_montly_hours'],
    y=left_employees['satisfaction_level'],
    alpha=0.5, c='red', label='Left'
)
plt.xlabel("Average Monthly Hours")
plt.ylabel("Satisfaction Level")
plt.title("Satisfaction vs Hours by Exit Status")
plt.legend()
plt.show()

plt.scatter(
    x=retained['average_montly_hours'],
    y=retained['satisfaction_level'],
    alpha=0.5, c='blue', label='Stayed'
)
plt.xlabel("Average Monthly Hours")
plt.ylabel("Satisfaction Level")
plt.title("Satisfaction vs Hours by Exit Status")
plt.legend()
plt.show()

# Understand the effect of satisfaction level, department, promotion in last 5 years and salary level of employees who have left the organization.

mean_values.plot(kind='bar',figsize=(10,7))
plt.title('Mean Values of Factors by Retention Status')
plt.xlabel('Retention Status (0 = Stayed, 1 = Left)')
plt.ylabel('Mean Values')
plt.xticks(rotation=0)
plt.legend(title='Factors')
plt.tight_layout()
plt.show()

new_dataset = normalized_HR_data_df
Dept_count = new_dataset.groupby(['Department','left']).size().unstack(fill_value=0)
Dept_count.plot(kind='bar',figsize=(10,7))
plt.xlabel("Dept")
plt.ylabel("count")
plt.tight_layout()
plt.xticks(rotation=45)
plt.legend(title="Exit Status", labels = ['Stayed','left'])
plt.show()

promo_count = new_dataset.groupby(['promotion_last_5years','left']).size().unstack(fill_value=0)
promo_count.plot(kind='bar',figsize=(10,7))
plt.xlabel("promotion_last_5years")
plt.ylabel("count")
plt.tight_layout()
plt.xticks(rotation=45)
plt.legend(title="Exit Status", labels=['Stayed',"Left"])
plt.show()

sal_count = new_dataset.groupby(['salary','left']).size().unstack(fill_value=0)
sal_count.plot(kind='bar', figsize=(10,7))
plt.xlabel('salary')
plt.ylabel('count')
plt.xticks(rotation=45)
plt.legend(title='Exit Survey', labels=['Stayed','Left'])
plt.tight_layout()
plt.show()

#................the method that we see below is not a clean approach of solving this problem. as it well lead to duplication of many columns. both df and normalized_HR_data_df have numeric columns and by concatenating you’ll have two copies of the numeric columns. That means merged3 will not be clean, and might still have leftover categorical values if not dropped correctly..........#



# dummies1 = pd.get_dummies(normalized_HR_data_df.Department)
# dummies1 = dummies1.astype(int)
# merged1 = pd.concat([dummies1,normalized_HR_data_df],axis="columns")
# final1 = merged1.drop(['Department','hr'],axis=1)
# print(final1)
#
# normalized_HR_data_df = normalized_HR_data_df.join(df['salary'])
# print(normalized_HR_data_df['salary'])
#
# dummies2 = pd.get_dummies(df.salary)
# dummies2 = dummies2.astype(int)
# merged2 = pd.concat([df,dummies2],axis='columns')
# final2 = merged2.drop(['salary','high'],axis='columns')
# print(final2)
#
# merged3 = pd.concat([final1, final2], axis='columns')
# print(merged3)


dept_dummies = pd.get_dummies(df['Department'],drop_first=True)
salary_dummies = pd.get_dummies(df['salary'],drop_first=True)

X = pd.concat([normalized_HR_data_df.drop(['Department',"salary"], axis=1),dept_dummies,salary_dummies],axis=1)
y = df['left']

print(f"final feature matrix shape {X.shape}")
print(f"target matrix shape {y.shape}")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8)

# now here we have a choice.....and either of the following can be used for this classification problem.....LogisticRegression, SVC, RandomForestClassifier, AdaBoostClassifier...by training each we can check the score and whosoever gives tge best score will be suitable for our model, ANN

#............................LogisticRegression...............................#

from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression(max_iter=100)
print(model1.fit(X_train,y_train))
print(f"train accuracy of LogisticRegression {model1.score(X_train,y_train)}")
print(f"test accuracy of LogisticRegression {model1.score(X_test,y_test)}")

from sklearn.svm import SVC
model2 = SVC(max_iter=100)
print(model2.fit(X_train,y_train))
print(f"train accuracy of SVC {model2.score(X_train,y_train)}")
print(f"test accuracy of SVC {model2.score(X_test,y_test)}")

from sklearn.ensemble import RandomForestClassifier
model3 = RandomForestClassifier(n_estimators=10, min_samples_split=10, min_impurity_decrease=0.001, max_depth=7)
print(model3.fit(X_train,y_train))
print(f"train accuracy of RandomForestClassifier {model3.score(X_train,y_train)}")
print(f"test accuracy of RandomForestClassifier {model3.score(X_test,y_test)}")

from sklearn.ensemble import AdaBoostClassifier
model4 = AdaBoostClassifier(n_estimators=10)
print(model4.fit(X_train,y_train))
print(f"train accuracy of AdaBoostClassifier {model4.score(X_train,y_train)}")
print(f"test accuracy of AdaBoostClassifier {model4.score(X_test,y_test)}")

from sklearn.neural_network import MLPClassifier
model5 = MLPClassifier(hidden_layer_sizes=50)
print(model5.fit(X_train,y_train))
print(f"train accuracy of MLPClassifier {model5.score(X_train,y_train)}")
print(f"test accuracy of MLPClassifier {model5.score(X_test,y_test)}")


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model1, X, Y, cv=5)
print("LogisticRegression CV scores:", scores)
print("Mean CV accuracy:", scores.mean())

#...................Check class balance....If left=1 vs left=0 is very unbalanced (e.g., 90% stayed, 10% left), accuracy might be misleading.

print(Y.value_counts(normalize=True))

#............................confusion_matrix.........................#

from sklearn.metrics import confusion_matrix
y_predicted = model4.predict(X_test)
cm = confusion_matrix(y_test, y_predicted)
print(cm)
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot = True )
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()

from sklearn.metrics import classification_report
print(classification_report(y_test,y_predicted))