import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

print("⏳ Loading creditcard.csv dataset...")
try:
    # Load dataset
    df = pd.read_csv("creditcard.csv")
    
    # Feature columns (30 in total: Time, V1-V28, Amount)
    X = df.drop('Class', axis=1)
    
    # Target variable (0 = Legitimate, 1 = Fraudulent)
    y = df['Class']

    print(f"✅ Data loaded successfully. Shape: {df.shape}")
    print(f"📊 Class distribution:\n{y.value_counts()}")

    # Split into training and test sets
    print("⏳ Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("⏳ Training the Logistic Regression model (with class_weight='balanced')...")
    # We must handle the severe class imbalance (very few frauds)
    # class_weight='balanced' automatically adjusts weights inversely proportional to class frequencies.
    # We also add a StandardScaler because the raw data has different scales (e.g. Amount vs V-features)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)

    print("⏳ Saving the model...")
    # Save the trained pipeline properly
    with open("logistic_model.pkl", "wb") as file:
        pickle.dump(pipeline, file)

    print("✅ Model trained and saved successfully as 'logistic_model.pkl'!")

except FileNotFoundError:
    print("❌ Error: 'creditcard.csv' not found in the current directory.")
except Exception as e:
    print(f"❌ An error occurred: {e}")
