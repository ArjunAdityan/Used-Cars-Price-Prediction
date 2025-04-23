This project focuses on predicting the prices of used cars using machine learning techniques, based on the dataset provided in the Cohort-4 KaggleX Skill Assessment Challenge (May 2024).

📁 Project Overview
The repository includes:

Preprocessing & Feature Engineering: Conducted on the encoded_test dataset and stored in kaggle_cohort_test.py.

Model Training & Evaluation: Performed on the encoded_train dataset with analysis of various machine learning models in kaggle_cohort_submission.py.

📊 Model Performance
Among the various models evaluated, the Bagged Random Forest showed the best performance on the training dataset. However, on the test dataset, the Voting Regressor (combining Bagged XGBoost and Bagged Random Forest) outperformed individual models and secured 1st place in the KaggleX Challenge leaderboard.

🌐 Web Deployment
The final Voting Regressor model is deployed as a web application using Flask, HTML, and CSS.

app.py – Backend logic for the Flask application.

templates/index.html – Frontend interface for user interaction.

📦 Files and Directories

File/Directory	Description
kaggle_cohort_test.py	Preprocessing and prediction pipeline for test set
kaggle_cohort_submission.py	Model training, evaluation, and ensemble creation
app.py	Flask application backend
templates/index.html	Frontend of the web application
🏆 Challenge Recognition
This project secured Rank #1 in the Cohort-4 KaggleX Skill Assessment Challenge (May 2024).
