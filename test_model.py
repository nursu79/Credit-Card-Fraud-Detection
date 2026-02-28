import pickle

# Try loading the model
try:
    with open("logistic_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully!")
    print("Model type:", type(model))
except Exception as e:
    print("❌ Error loading model:", e)
