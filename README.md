# 📡 Telecom Customer Churn Prediction

A machine learning web application that predicts whether a telecom customer is likely to churn, built with an Artificial Neural Network (ANN) and deployed using Streamlit.

🔗 **Live App**: https://ann-telcome-customer-churn-analysis-j7ypkn7w7pcm3d5ek3otbi.streamlit.app/
**Screenshot**: <img width="1806" height="860" alt="image" src="https://github.com/user-attachments/assets/ed209018-45d4-4538-99cb-ed5f410747ac" />

---

## 📌 Project Overview

Customer churn is one of the biggest challenges in the telecom industry. This project builds an end-to-end machine learning pipeline — from data preprocessing to a deployed web app — that identifies at-risk customers so the business can take proactive retention action.

---

## 📊 Dataset

- **Source**: Telco Customer Churn Dataset
- **Records**: 7,043 customers
- **Features**: 20+ features including demographics, services subscribed, contract type, billing, and tenure
- **Target**: Churn Label (Yes / No)
- **Class imbalance**: ~73% No Churn / ~27% Churn

---

## 🧠 Model Architecture

| Layer | Units | Activation | Regularization |
|---|---|---|---|
| Input | (n_features,) | — | — |
| Dense | 32 | ReLU | L2 + BatchNorm + Dropout(0.4) |
| Dense | 16 | ReLU | L2 + BatchNorm + Dropout(0.3) |
| Dense | 8 | ReLU | L2 + Dropout(0.3) |
| Output | 1 | Sigmoid | — |

**Training setup:**
- Optimizer: Adam (lr = 0.0005)
- Loss: Binary Crossentropy
- Class weights used to handle class imbalance
- EarlyStopping (patience=8) + ReduceLROnPlateau
- 3-way split: Train (70%) / Validation (15%) / Test (15%)

---

## 📈 Model Performance

| Metric | Score |
|---|---|
| Test Accuracy | ~0.75 |
| Test ROC-AUC | ~0.82 |
| Churn Recall | ~0.75 |
| Churn Precision | ~0.52 |
| Churn F1-Score | ~0.62 |

> **Note**: The decision threshold is tuned on the validation set (not the test set) to maximise F1-score on the churn class. Default 0.5 is replaced with a tuned threshold (~0.49) to improve recall — catching more at-risk customers at a small cost to precision.

---

## 🗂️ Project Structure

```
Customer_chun_ANN/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── ANN_project_telcom_churn.ipynb  # Full notebook (EDA + preprocessing + model)
├── Telco_customer_churn.xlsx       # Raw dataset
│
└── artifacts/
    ├── churn_model.onnx            # Trained ANN (ONNX format for deployment)
    ├── preprocessor.joblib         # Fitted ColumnTransformer (scaler + encoder)
    ├── label_encoder.joblib        # Fitted LabelEncoder for target
    └── metadata.json               # Threshold + feature columns + test metrics
```

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.x |
| ML Framework | TensorFlow / Keras |
| Preprocessing | Scikit-learn (StandardScaler, OneHotEncoder, ColumnTransformer) |
| Model Export | ONNX (via tf2onnx) |
| Model Inference | ONNX Runtime |
| Web App | Streamlit |
| Deployment | Streamlit Cloud |
| Version Control | Git + GitHub |

---

## 🚀 Run Locally

**1. Clone the repository:**
```bash
git clone https://github.com/S2023-maker/ANN-Telcome-Customer-Churn-Analysis.git
cd ANN-Telcome-Customer-Churn-Analysis
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run the app:**
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 🔍 How It Works

1. User enters customer details (demographics, services, contract, billing) into the web form
2. Input is passed through the saved `preprocessor.joblib` (same scaling/encoding as training)
3. Processed features are fed into the ONNX model for inference
4. The predicted probability is compared against the tuned threshold
5. Result is displayed as **High Churn Risk** ⚠️ or **Low Churn Risk** ✅ with contributing risk factors

---

## 🛡️ Deployment Notes

- Model is exported to ONNX format to avoid TensorFlow dependency on Streamlit Cloud
- Threshold tuned on validation set only — test set touched exactly once for final evaluation
- Preprocessing pipeline saved as a joblib artifact to ensure identical feature transformation at inference time

---

## 👩‍💻 Author

**Shreya R Joshi**  
[GitHub](https://github.com/S2023-maker)
