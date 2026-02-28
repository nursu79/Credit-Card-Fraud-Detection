# SecureTransaction AI - Credit Card Fraud Detection

A Machine Learning-powered web application built with Python and Flask that analyzes credit card transactions in real-time to detect fraudulent activity.

## What This Project Does

This application provides a modern, fast web interface where you can submit the anonymized features of a credit card transaction (Principal Components: `V1` to `V28`, plus `Time` and `Amount`). The backend then runs those 30 values against a trained Logistic Regression machine learning model.

If the input features match the complex mathematical patterns associated with fraud, the system immediately flags the transaction as **"Fraudulent" (⚠️)**. Otherwise, it approves it as **"Legitimate" (✅)**.

## How to Run it Locally

### Prerequisites
* Python 3.8+ installed on your computer.

### Installation Steps

1. **Clone or Download** the project folder.
2. **Open your terminal** and navigate into the project directory:
   ```bash
   cd "credit project - Copy/credit project - Copy"
   ```
3. **Create a fresh virtual environment** (if you haven't already):
   ```bash
   python3 -m venv venv
   ```
4. **Activate the environment**:
   * *Linux / macOS:* `source venv/bin/activate`
   * *Windows:* `venv\Scripts\activate`
5. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
6. **Start the Flask Server**:
   ```bash
   python3 app.py
   ```
7. **Access the App**: Open your browser and go to `http://127.0.0.1:5000/`.

---

## 🧠 Step-by-Step: The Machine Learning Training Process

If you want to train the model from scratch on new data, you can run `python3 train_model.py`. Here is exactly what is happening in that script:

### 1. The Dataset
The model is trained on `creditcard.csv` (not included in the repo due to its large 150MB size). This is a highly imbalanced dataset containing 284,807 European credit card transactions from September 2013, where **only 492 transactions are fraudulent (0.17%)**.

### 2. Loading & Preprocessing
The script splits the data into 30 feature columns (`X`) and the target `Class` column (`y`). It then uses `train_test_split` to reserve 20% of the data exclusively for testing.

### 3. Creating the Pipeline
To address the severe class imbalance and the varying scales of the data (the `Amount` column varies wildly compared to the PCA `V` features), we wrap the model in a Scikit-Learn `Pipeline`:
* **`StandardScaler()`**: Normalizes all 30 features so they have a mean of `0` and a standard deviation of `1`. This significantly improves Logistic Regression stability.
* **`LogisticRegression()`**: The actual predictive algorthm.
    * **Crucial Step:** We pass `class_weight='balanced'` explicitly. This forces the model to heavily penalize errors made on the rare Fraudulent class by assigning it a mathematically higher weight inversely proportional to its frequency. Without this, the model would simply guess "Legitimate" 100% of the time.

### 4. Saving the Model
Finally, the entire pipeline (including the scaler and the regressor) is serialized and saved as `logistic_model.pkl` using Python's `pickle` library, allowing the Flask `app.py` to load and use it instantly.

---

## 🚀 Deployment Instructions (Render or Heroku)

This project has already been configured for cloud deployment with a `Procfile` and a `requirements.txt` containing the `gunicorn` production server.

### Deploying to Render.com (Free and easiest)

1. **GitHub Setup:**
   * Open the `.gitignore` file.
   * **Remove** the line `logistic_model.pkl` from the bottom. (Your deployment server *needs* the trained model file to run predictions).
   * Commit and push your code to a new GitHub repository.

2. **Render Setup:**
   * Go to [Render.com](https://render.com/) and connect your GitHub account.
   * Click **New +** > **Web Service**.
   * Select your new repository.

3. **Configure the Service:**
   * **Name:** `secure-transaction-ai`
   * **Language:** `Python 3`
   * **Build Command:** `pip install -r requirements.txt`
   * **Start Command:** `gunicorn app:app` (This uses the `Procfile`)

4. **Deploy:** Click **Create Web Service**. Within a few minutes, Render will build your app and provide you with a live, shareable URL!
