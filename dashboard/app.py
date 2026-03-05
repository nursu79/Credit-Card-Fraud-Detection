import streamlit as st
import requests
import json
import os
import sys
import pandas as pd
from sqlalchemy.orm import Session

# Allow importing database module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db import SessionLocal, engine, Base
from database.models import Transaction

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)

# Page Config
st.set_page_config(page_title="Fraud Detection Dashboard", page_icon="🛡️", layout="wide")

st.title("🛡️ AI Fraud Detection & Investigation Platform")

# FastAPI endpoint URLs
API_URL = os.getenv("API_URL", "http://localhost:8000")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["Analyze Transaction", "Recent Alerts", "System Stats"])

# --- Tab 1: Analyze Transaction ---
with tab1:
    st.header("Submit Transaction for Analysis")
    st.markdown("Enter transaction details manually, parse raw data, or use samples.")
    
    # --- Bulk Paste Feature ---
    with st.expander("📥 Bulk Paste Raw Transaction Data (CSV Format)", expanded=False):
        raw_data = st.text_area("Paste 30 features (Time, V1...V28, Amount) separated by commas:", 
                               placeholder="0.0, -2.31, 1.95, ...")
        if st.button("Parse & Load Data"):
            try:
                values = [float(x.strip()) for x in raw_data.split(",")]
                if len(values) >= 30:
                    st.session_state['Time'] = values[0]
                    for i in range(1, 29):
                        st.session_state[f'V{i}'] = values[i]
                    st.session_state['Amount'] = values[29]
                    st.success("✅ Data parsed and loaded successfully!")
                else:
                    st.error(f"❌ Expected 30 values, but found {len(values)}. Please check your data.")
            except ValueError:
                st.error("❌ Invalid format. Please ensure all values are numbers separated by commas.")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Load Full Anomalous Sample (Real Fraud)"):
            # Sample: 406,-2.3122265423263,1.95199201064158,-1.60985073229769,3.9979055875468,-0.522187864667764,-1.42654531920595,-2.53738730624579,1.39165724829804,-2.77008927719433,-2.77227214465915,3.20203320709635,-2.89990738849473,-0.595221881324605,-4.28925378244217,0.389724120274487,-1.14074717980657,-2.83005567450437,-0.0168224681808257,0.416955705037907,0.126910559061474,0.517232370861764,-0.0350493686052974,-0.465211076182388,0.320198198514526,0.0445191674731724,0.177839798284401,0.261145002567677,-0.143275874698919,0
            sample_vals = [406.0, -2.312226, 1.951992, -1.609850, 3.997905, -0.522187, -1.426545, -2.537387, 1.391657, -2.770089, -2.772272, 3.202033, -2.899907, -0.595221, -4.289253, 0.389724, -1.140747, -2.830055, -0.016822, 0.416955, 0.126910, 0.517232, -0.035049, -0.465211, 0.320198, 0.044519, 0.177839, 0.261145, -0.143275, 0.0]
            st.session_state['Time'] = sample_vals[0]
            for i in range(1, 29):
                st.session_state[f'V{i}'] = sample_vals[i]
            st.session_state['Amount'] = sample_vals[29]
            st.rerun()

    # --- Core Inputs ---
    col1, col2, col3 = st.columns(3)
    with col1:
        time_val = st.number_input("Time", value=st.session_state.get('Time', 0.0), format="%.2f", step=1.0)
    with col2:
        amount_val = st.number_input("Amount ($)", value=st.session_state.get('Amount', 45.0), format="%.2f", step=0.01)

    st.markdown("---")
    st.subheader("PCA Features (V1 - V28)")
    
    # Display V1-V28 in a grid
    v_features = {}
    cols = st.columns(4)
    for i in range(1, 29):
        with cols[(i-1) % 4]:
            key = f"V{i}"
            v_features[key] = st.number_input(f"Feature {key}", value=st.session_state.get(key, 0.0), format="%.6f", step=0.000001)

    # Pack up all features
    tx_data = {"Time": time_val, "Amount": amount_val}
    tx_data.update(v_features)
    
    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Predict Fraud (Fast)", use_container_width=True):
            try:
                res = requests.post(f"{API_URL}/predict", json=tx_data)
                res.raise_for_status()
                data = res.json()
                if data['risk_level'] == "HIGH":
                    st.error(f"🔴 Risk Level: {data['risk_level']}")
                elif data['risk_level'] == "MEDIUM":
                    st.warning(f"🟡 Risk Level: {data['risk_level']}")
                else:
                    st.success(f"🟢 Risk Level: {data['risk_level']}")
                st.metric("Probability of Fraud", f"{data['fraud_probability']:.2%}")
            except Exception as e:
                st.error(f"API Request Failed: {e}")

    with col_b:
        if st.button("Investigate (RAG AI Analysis)", type="primary", use_container_width=True):
            try:
                with st.spinner("Analyzing transaction and generating AI explanation..."):
                    res = requests.post(f"{API_URL}/investigate", json=tx_data)
                    res.raise_for_status()
                    data = res.json()
                    
                    if data['risk_level'] == "HIGH":
                        st.error(f"🔴 Risk Level: {data['risk_level']}")
                    elif data['risk_level'] == "MEDIUM":
                        st.warning(f"🟡 Risk Level: {data['risk_level']}")
                    else:
                        st.success(f"🟢 Risk Level: {data['risk_level']}")
                        
                    st.metric("Probability of Fraud", f"{data['fraud_probability']:.2%}")
                    
                    st.markdown("### 🤖 AI Investigation Explanation")
                    st.info(data['ai_explanation'])
                    
                    st.markdown("### 🔍 Detected Relevant Patterns")
                    for pattern in data['fraud_patterns']:
                        st.markdown(f"- **{pattern}**")
                        
                    st.markdown("### 🛡️ Recommended Action")
                    st.warning(data['recommended_action'])
                    
            except Exception as e:
                st.error(f"API Request Failed: {e}")


# --- Tab 2: Recent Alerts ---
with tab2:
    st.header("Recent Transaction Investigations")
    
    try:
        db: Session = SessionLocal()
        # Fetch latest 20
        recent = db.query(Transaction).order_by(Transaction.timestamp.desc()).limit(20).all()
        db.close()
        
        if recent:
            records = [{
                "Time": t.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "Amount": t.Amount,
                "Risk Level": t.risk_level,
                "Fraud Probability": f"{t.fraud_probability:.2%}",
                "Explanation Preview": (t.investigation_explanation[:100] + "...") if t.investigation_explanation else ""
            } for t in recent]
            
            df = pd.DataFrame(records)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No investigations performed yet.")
    except Exception as e:
        st.warning(f"Could not connect to database: {e}")

# --- Tab 3: System Stats ---
with tab3:
    st.header("System Subcomponents Status")
    
    try:
        res = requests.get(f"{API_URL}/")
        if res.status_code == 200:
            st.success("✅ FastAPI Backend is Online")
        else:
            st.error("❌ FastAPI Backend is Offline")
    except:
        st.error("❌ FastAPI Backend is Offline")
        
    st.info("🧠 Machine Learning Model: Loaded (LogisticRegression)")
    st.info("📚 Vector Database: Loaded (FAISS with MiniLM)")
    st.info("🗄️ Database: PostgreSQL (Docker) / SQLite (Local)")
