# 🚗 Used Car Price Prediction

> **1st Place Winner** - Cohort-4 KaggleX Skill Assessment Challenge (May 2024)

## 📋 Project Overview

This project applies machine learning techniques to predict used car prices using the dataset from the Cohort-4 KaggleX Skill Assessment Challenge. By implementing advanced preprocessing, feature engineering, and ensemble modeling approaches, the solution achieved top performance on the competition leaderboard.

## 🏗️ Project Structure

- **Preprocessing & Feature Engineering**: Implemented in `kaggle_cohort_test.py` for the encoded_test dataset
- **Model Training & Evaluation**: Conducted in `kaggle_cohort_submission.py` using the encoded_train dataset
- **Web Application**: Deployed using Flask, HTML, and CSS

## 📊 Model Performance

The model evaluation revealed:

- **Best Training Performance**: Bagged Random Forest
- **Best Test Performance**: Voting Regressor (combining Bagged XGBoost and Bagged Random Forest)

The final ensemble model secured **1st place** in the KaggleX Challenge leaderboard.

## 💻 Web Application

The solution includes a user-friendly web interface that allows users to predict car prices by inputting vehicle specifications.

**Tech Stack:**
- Backend: Flask (Python)
- Frontend: HTML, CSS
- ML Pipeline: Scikit-learn, XGBoost

## 📁 Repository Structure

| File/Directory | Description |
|----------------|-------------|
| `kaggle_cohort_test.py` | Preprocessing and prediction pipeline for the test dataset |
| `kaggle_cohort_submission.py` | Model training, evaluation, and ensemble creation |
| `app.py` | Flask application backend logic |
| `templates/index.html` | Frontend interface for the web application |

## 🚀 Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the web application: `python app.py`
4. Access the interface at `http://localhost:5000`

## 📈 Key Insights

- Feature importance analysis revealed that vehicle age, mileage, and brand were the strongest predictors
- Ensemble methods consistently outperformed individual models
- Bagging techniques helped reduce overfitting and improve generalization

## 🏆 Recognition

This project ranked **#1** in the Cohort-4 KaggleX Skill Assessment Challenge (May 2024), demonstrating superior performance in both prediction accuracy and implementation.
