
This project is a supervised machine learning solution that predicts whether a passenger survived the Titanic disaster.
It uses the Kaggle Titanic dataset and includes the full pipeline of a real-world ML problem:

✅ Key Features:
Data Cleaning: Handled missing values (Age, Embarked, etc.) and extracted meaningful features (like Cabin letter).

Feature Engineering: Converted categorical columns to numerical using Label Encoding and One-Hot Encoding.

Exploratory Data Analysis (EDA): Analyzed patterns between features like Sex, Pclass, Age, Fare and survival.

Modeling: Trained classification models like:

RandomForestClassifier

(optionally) Logistic Regression, Decision Trees, etc.

Evaluation:

Achieved ~88% accuracy on internal validation.

Scored 0.76076 on Kaggle's public leaderboard.

Deployment-ready pipeline: Custom functions to clean and prepare data consistently for train/test splits or external test files.
