import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

url = "https://raw.githubusercontent.com/AnjulaMehto/Sampling_Assignment/main/Creditcard_data.csv"
data = pd.read_csv(url)

X = data.drop('Class', axis=1)
y = data['Class']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
balanced_data = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled, columns=['Class'])], axis=1)


sample_size = int(len(balanced_data) * 0.2)  
samples = [balanced_data.sample(sample_size, random_state=i) for i in range(5)]


sampling_techniques = {
    'Sampling1': lambda x: train_test_split(x.drop('Class', axis=1), x['Class'], test_size=0.2, random_state=42),
    'Sampling2': lambda x: train_test_split(x.drop('Class', axis=1), x['Class'], test_size=0.3, stratify=x['Class'], random_state=42),
    'Sampling3': lambda x: train_test_split(x.drop('Class', axis=1), x['Class'], test_size=0.25, random_state=42),
    'Sampling4': lambda x: train_test_split(x.drop('Class', axis=1), x['Class'], test_size=0.2, stratify=x['Class'], random_state=42),
    'Sampling5': lambda x: train_test_split(x.drop('Class', axis=1), x['Class'], test_size=0.3, random_state=42)
}

sampled_data = {}
for i, sample in enumerate(samples):
    sampling_name = f'Sampling{i+1}'
    sampled_data[sampling_name] = sampling_techniques[sampling_name](sample)


models = {
    'M1': LogisticRegression(random_state=42),
    'M2': DecisionTreeClassifier(random_state=42),
    'M3': RandomForestClassifier(random_state=42),
    'M4': SVC(random_state=42),
    'M5': KNeighborsClassifier()
}

results = {}

for model_name, model in models.items():
    for sampling_name, (X_train, X_test, y_train, y_test) in sampled_data.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[(model_name, sampling_name)] = accuracy


results_df = pd.DataFrame(list(results.items()), columns=['Model_Sampling', 'Accuracy'])
results_df[['Model', 'Sampling']] = pd.DataFrame(results_df['Model_Sampling'].tolist(), index=results_df.index)
results_df = results_df.pivot(index='Model', columns='Sampling', values='Accuracy')

print("Accuracy Results:")
print(results_df)


results_df.to_csv('sampling_results.csv')
