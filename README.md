# Student-Exam-Score-Predictor
# 🎓 Student Exam Score Predictor 📊

This project explores how **student lifestyle habits** impact their academic performance, using a simulated dataset from Kaggle. The goal is to build a **machine learning model** that predicts a student's final exam score based on features like study hours, sleep patterns, social media use, mental health, and more.

[🔗 Dataset on Kaggle](https://www.kaggle.com/datasets/jayaantanaath/student-habits-vs-academic-performance)

---

## 📁 Project Structure

- `Student Exam Scores Prediction.ipynb` – Full notebook with EDA, preprocessing, model training, evaluation, and model saving.
- `app.py` – Streamlit web app for deploying the prediction model.
- `README.md` – Project overview and documentation.
- Screanshopt of the webb app

---

## 📦 Dataset Overview

The dataset titled **"Student Habits vs Academic Performance"** contains:
- 1,000 synthetic student records
- lifestyle features such as:
  - Study hours per day
  - Sleep hours per night
  - Mental health rating
  - Social media usage
  - Diet quality
  - Attendance percentage
  - Final exam scores (target)

> _“Ever wondered how much Netflix, sleep, or TikTok scrolling affects your grades?”_ 👀  
This dataset simulates real student behavior for data analysis and ML experiments. It’s perfect for:
- Exploratory Data Analysis (EDA)
- Regression and prediction modeling
- Data visualization
- Educational case studies

---

## 🧠 Models Trained

Three machine learning models were trained and evaluated:
1. **Linear Regression**
2. **Decision Tree Regressor**
3. **Random Forest Regressor**

### 🎯 Model Evaluation Metric:
- **Root Mean Squared Error (RMSE)**
- **R² score**

### ✅ Best Performing Model:
> **Linear Regression** had the **lowest RMSE and the highest R² score**, indicating

The smallest average prediction error

The strongest overall fit to the data

This made it the most reliable and interpretable model for predicting student exam scores.
---

## 🚀 Streamlit App

A lightweight web app was built using **Streamlit** where users can input lifestyle data and get an instant prediction of exam score.

### 🔧 Features:
- Input sliders and dropdowns for study hours, attendance, mental health, sleep, part-time job, etc.
- Live prediction using the trained Linear Regression model.
- SHAP visualizations for interpreting feature contributions.

