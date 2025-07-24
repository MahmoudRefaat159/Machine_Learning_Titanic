import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

le =LabelEncoder()
encoders = {}



def show_all_label_mappings(encoders):
    for col, le in encoders.items():
        print(f"\n Models: {col}")
        for i, cls in enumerate(le.classes_):
            print(f"  {i} => {cls}")


train_data = pd.read_csv('train.csv')

test_data = pd.read_csv('test.csv')


def clean(df, is_train=True):
    df['Cabin'] = df['Cabin'].notna().astype(int)
    df['Age'] = SimpleImputer(strategy='mean').fit_transform(df[['Age']])
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    obj_cols = df.select_dtypes(include='object').columns
    for col in obj_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

for col in train_data:
    le = LabelEncoder()
    train_data [col] = le.fit_transform(train_data [col])
    encoders[col] = le

x = clean(train_data)[[col for col in train_data.columns if col != 'Survived']]
y=clean(train_data)['Survived']

test_data=clean(test_data)



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=42)



model = RandomForestClassifier(n_estimators=300, random_state=42,max_depth=7)
model.fit(x_train, y_train)




y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f" {accuracy * 100:.2f}%")

test_cleaned = clean(test_data, is_train=False)
test_predictions =model.predict(test_cleaned)

submission = pd.DataFrame({
   'PassengerId': test_data['PassengerId'],
    'Survived': test_predictions
})

submission.to_csv('titanic_predictions.csv', index=False)

