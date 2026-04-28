# 🚀 Employee Attrition Prediction

## 📌 Overview
This project predicts employee attrition (whether an employee will leave the company) using machine learning techniques.  
It uses the IBM HR Analytics dataset and applies data preprocessing, feature engineering, and classification models.

---

## 🎯 Objective
- Analyze employee data to identify attrition patterns  
- Perform data cleaning and preprocessing  
- Build machine learning models to predict attrition  
- Identify key factors affecting employee turnover  

---

## 📂 Project Structure
```
employee-attrition-prediction/
│
├── data/
│   └── attrition.csv
│
├── src/
│   └── attrition_model.py
│
├── requirements.txt
├── README.md
```

---

## ⚙️ Technologies Used
- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

## 🔍 Data Preprocessing
- Removed duplicate records  
- Converted Attrition column into numeric (Yes → 1, No → 0)  
- Dropped unnecessary columns (EmployeeNumber, Over18, etc.)  
- Applied one-hot encoding for categorical variables  
- Feature scaling using StandardScaler  

---

## 🤖 Models Used
- Logistic Regression  
- Random Forest  

---

## 📊 Model Evaluation

### Logistic Regression
- ROC-AUC: **0.81**

### Random Forest
- Accuracy: **84%**
- ROC-AUC: **0.79**

### Performance Insight
- Model predicts non-attrition very well  
- Lower recall for attrition class (imbalanced dataset)

---

## 📈 Top Features
- Monthly Income  
- Age  
- Total Working Years  
- Years at Company  
- Daily Rate  
- Hourly Rate  

---

## 📌 Key Insights
- Lower income employees are more likely to leave  
- Experience and tenure reduce attrition  
- Work-related factors influence employee retention  
- Dataset is imbalanced → harder to predict attrition cases  

---

## ⚙️ How to Run

1. Clone repository:
```bash
git clone https://github.com/your-username/employee-attrition-prediction.git
```

2. Go to folder:
```bash
cd employee-attrition-prediction
```

3. Install libraries:
```bash
pip install -r requirements.txt
```

4. Run project:
```bash
python src/attrition_model.py
```

---

## 🚀 Future Improvements
- Handle class imbalance (SMOTE / class weighting)  
- Hyperparameter tuning  
- Add advanced models (XGBoost)  
- Build dashboard or web app  

---

## 👨‍💻 Author
Karthikeyan
