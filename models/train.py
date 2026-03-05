import os
import sys
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# Add parent directory to path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_pipeline import DataPipeline

def train_and_evaluate():
    print("⏳ Loading dataset...")
    try:
        pipeline = DataPipeline("creditcard.csv")
        df = pipeline.load_data()
        
        print(f"✅ Data loaded successfully. Shape: {df.shape}")
        
        print("⏳ Splitting data into train/test sets...")
        X_train, X_test, y_train, y_test = pipeline.preprocess_and_split(df)
        
        print("⏳ Training Logistic Regression model (with class_weight='balanced')...")
        model_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
        ])
        
        model_pipeline.fit(X_train, y_train)
        
        print("⏳ Evaluating the model on test set...")
        y_pred = model_pipeline.predict(X_test)
        y_prob = model_pipeline.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        print("\n=== Model Performance ===")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f} (Crucial for minimizing false positives)")
        print(f"Recall   : {recall:.4f} (Crucial for catching frauds)")
        print(f"F1-Score : {f1:.4f}")
        print(f"ROC-AUC  : {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("⏳ Saving the model...")
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", "logistic_model.pkl")
        with open(model_path, "wb") as file:
            pickle.dump(model_pipeline, file)
            
        print(f"✅ Model trained and saved successfully at '{model_path}'!")
        
    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    train_and_evaluate()
