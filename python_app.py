import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class CreditRiskModel:
    def __init__(self):
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.scaler = StandardScaler()
        self.features = ['age', 'income', 'loan_amount', 'delinquent_months', 
                        'total_dpd', 'credit_utilization_ratio', 'income_loan_ratio']
        self.risk_labels = ['Poor', 'Average', 'Good']
        
    def load_and_prepare_data(self):
        print("Loading datasets...")
        
        customers = pd.read_csv("customers.csv")
        loans = pd.read_csv("loans.csv")
        bureau = pd.read_csv("bureau_data.csv")
        
        print(f"Customers: {customers.shape}, Loans: {loans.shape}, Bureau: {bureau.shape}")
        
        df = customers.merge(loans, on='cust_id').merge(bureau, on='cust_id')
        print(f"Merged dataset shape: {df.shape}")
        
        df['income_loan_ratio'] = df['income'] / (df['loan_amount'] + 1)
        df['risk_score'] = (100 - df['delinquent_months'] * 3 - 
                           df['total_dpd'] * 0.2 - df['credit_utilization_ratio'] * 0.5)
        
        df['risk_class'] = pd.cut(df['risk_score'], 
                                 bins=[-np.inf, 40, 70, np.inf], 
                                 labels=[0, 1, 2]).astype(int)
        
        print(f"Risk distribution:\n{df['risk_class'].value_counts().sort_index()}")
        return df
    
    def train_and_evaluate(self):
        df = self.load_and_prepare_data()
        
        X = df[self.features].fillna(df[self.features].median())
        y = df['risk_class']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Training Logistic Regression model...")
        self.model.fit(X_train_scaled, y_train)
        
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\n=== MODEL PERFORMANCE ===")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.risk_labels))
        
        self.plot_confusion_matrix(cm)
        self.plot_feature_importance()
        
        return cm, accuracy, X_test, y_test, y_pred
    
    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.risk_labels, yticklabels=self.risk_labels)
        plt.title('Credit Risk Classification - Confusion Matrix')
        plt.xlabel('Predicted Risk Level')
        plt.ylabel('Actual Risk Level')
        plt.tight_layout()
        plt.show()
        
        print(f"\n=== CONFUSION MATRIX BREAKDOWN ===")
        for i, actual in enumerate(self.risk_labels):
            for j, predicted in enumerate(self.risk_labels):
                print(f"Actual {actual} → Predicted {predicted}: {cm[i][j]}")
    
    def plot_feature_importance(self):
        plt.figure(figsize=(12, 8))
        
        coefs = self.model.coef_
        x_pos = np.arange(len(self.features))
        width = 0.25
        
        for i, risk_level in enumerate(self.risk_labels):
            plt.bar(x_pos + i * width, coefs[i], width, label=f'{risk_level} Risk')
        
        plt.xlabel('Features')
        plt.ylabel('Coefficient Value')
        plt.title('Logistic Regression Coefficients by Risk Level')
        plt.xticks(x_pos + width, self.features, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def predict_risk(self, input_features):
        input_scaled = self.scaler.transform([input_features])
        prediction = self.model.predict(input_scaled)[0]
        probabilities = self.model.predict_proba(input_scaled)[0]
        
        risk_level = self.risk_labels[prediction]
        
        print(f"\n=== RISK ASSESSMENT RESULT ===")
        print(f"Risk Level: {risk_level}")
        print(f"Confidence: {max(probabilities):.1%}")
        print(f"\nProbability Breakdown:")
        for i, prob in enumerate(probabilities):
            print(f"  {self.risk_labels[i]}: {prob:.1%}")
        
        return risk_level, probabilities
    
    def assess_sample_applications(self):
        print(f"\n{'='*50}")
        print("SAMPLE LOAN APPLICATIONS ASSESSMENT")
        print(f"{'='*50}")
        
        samples = {
            "Low Risk Applicant": [35, 5000000, 2000000, 0, 0, 20, 2.5],
            "Medium Risk Applicant": [40, 2500000, 3000000, 3, 15, 50, 0.83],
            "High Risk Applicant": [25, 1000000, 4000000, 12, 60, 80, 0.25]
        }
        
        for applicant_type, features in samples.items():
            print(f"\n--- {applicant_type.upper()} ---")
            print(f"Age: {features[0]}, Income: {features[1]:,}, Loan: {features[2]:,}")
            print(f"Delinquent Months: {features[3]}, Days Past Due: {features[4]}")
            print(f"Credit Utilization: {features[5]}%")
            
            risk_level, probabilities = self.predict_risk(features)
            
            recommendations = {
                'Poor': 'REJECT - High default risk',
                'Average': 'CONDITIONAL APPROVAL - Higher interest rate recommended', 
                'Good': 'APPROVE - Standard terms applicable'
            }
            print(f"Recommendation: {recommendations[risk_level]}")

def create_web_interface():
    print(f"\n{'='*60}")
    print("INTERACTIVE CREDIT RISK ASSESSMENT")
    print(f"{'='*60}")
    
    try:
        print("\nPlease enter the following loan application details:")
        age = float(input("Age: "))
        income = float(input("Annual Income: "))
        loan_amount = float(input("Loan Amount: "))
        delinquent_months = float(input("Delinquent Months (0-24): "))
        total_dpd = float(input("Total Days Past Due: "))
        credit_utilization = float(input("Credit Utilization Ratio (0-100): "))
        
        income_loan_ratio = income / (loan_amount + 1)
        
        user_input = [age, income, loan_amount, delinquent_months, 
                     total_dpd, credit_utilization, income_loan_ratio]
        
        return user_input
        
    except ValueError:
        print("Invalid input. Using default values for demonstration.")
        return [35, 2500000, 2000000, 2, 10, 35, 1.25]

def main():
    print("CREDIT RISK ASSESSMENT MODEL")
    print("Using Logistic Regression with Confusion Matrix Analysis")
    print("-" * 60)
    
    model = CreditRiskModel()
    cm, accuracy, X_test, y_test, y_pred = model.train_and_evaluate()
    
    model.assess_sample_applications()
    
    user_features = create_web_interface()
    print(f"\nAssessing your application...")
    model.predict_risk(user_features)
    
    print(f"\n{'='*50}")
    print("MODEL STATISTICS SUMMARY")
    print(f"{'='*50}")
    print(f"Total test samples: {len(y_test)}")
    print(f"Correct predictions: {accuracy_score(y_test, y_pred) * len(y_test):.0f}")
    print(f"Model accuracy: {accuracy:.1%}")
    
    test_distribution = pd.Series(y_test).value_counts().sort_index()
    pred_distribution = pd.Series(y_pred).value_counts().sort_index()
    
    print(f"\nActual vs Predicted Distribution:")
    for i, risk in enumerate(model.risk_labels):
        actual = test_distribution.get(i, 0)
        predicted = pred_distribution.get(i, 0)
        print(f"{risk}: Actual={actual}, Predicted={predicted}")

# Execute the model
main()
