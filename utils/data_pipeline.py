import pandas as pd
from sklearn.model_selection import train_test_split

class DataPipeline:
    def __init__(self, data_path="creditcard.csv"):
        self.data_path = data_path

    def load_data(self):
        """Loads dataset from CSV."""
        try:
            df = pd.read_csv(self.data_path)
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: dataset '{self.data_path}' not found.")

    def preprocess_and_split(self, df, test_size=0.2, random_state=42):
        """Splits the dataset into training and test sets."""
        # Feature columns (30 in total: Time, V1-V28, Amount)
        X = df.drop('Class', axis=1)
        
        # Target variable (0 = Legitimate, 1 = Fraudulent)
        y = df['Class']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test
