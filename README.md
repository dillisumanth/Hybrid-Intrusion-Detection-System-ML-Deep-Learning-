# Hybrid Intrusion Detection System Using Machine Learning

## 📌 Project Overview

The **Hybrid Intrusion Detection System (IDS) Using Machine Learning** is an advanced cybersecurity solution designed to detect malicious network activities and classify attacks in real-time using Artificial Intelligence and Machine Learning techniques.

This project utilizes the **UNSW-NB15 dataset**, one of the most modern and realistic cybersecurity datasets, to train and evaluate machine learning models capable of identifying both normal and malicious network traffic patterns.

The system performs intelligent threat detection by analyzing network packet features and predicting potential cyberattacks such as:

* DoS (Denial of Service)
* Exploits
* Fuzzers
* Backdoors
* Reconnaissance
* Worms
* Shellcode Attacks
* Generic Attacks
* Analysis Attacks

The project provides a complete end-to-end workflow including:

* Data preprocessing
* Feature engineering
* Model training
* Real-time prediction
* Threat classification
* Scan result visualization
* Performance evaluation

It is developed using **Python**, **Machine Learning**, and modern data science libraries to demonstrate how AI can improve cybersecurity defense systems.

---

# 🚀 Features

## 🔍 Intelligent Threat Detection

Detects malicious traffic patterns using trained Machine Learning algorithms.

## 📊 Dataset-Based Learning

Uses the UNSW-NB15 dataset for realistic cybersecurity attack simulation.

## ⚡ Real-Time Prediction

Analyzes incoming network traffic and predicts whether it is normal or malicious.

## 🧠 Machine Learning Integration

Supports multiple ML algorithms such as:

* Random Forest
* Decision Tree
* Logistic Regression
* KNN
* SVM
* XGBoost (Optional)

## 📈 Performance Metrics

Displays:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

## 🖥️ Scan Visualization

Provides graphical outputs for attack detection and prediction analysis.

## 🔐 Cybersecurity Focused

Built specifically for:

* Security Research
* Intrusion Detection
* Threat Analysis
* Academic Projects
* Cybersecurity Learning

---

# 🛠️ Technologies Used

| Technology        | Purpose              |
| ----------------- | -------------------- |
| Python            | Core Development     |
| Pandas            | Data Processing      |
| NumPy             | Numerical Operations |
| Scikit-Learn      | Machine Learning     |
| Matplotlib        | Visualization        |
| Seaborn           | Data Analysis        |
| Joblib            | Model Saving         |
| UNSW-NB15 Dataset | Training Dataset     |

---

# 📂 Dataset Information

This project uses the **UNSW-NB15 Cybersecurity Dataset** developed by the Australian Centre for Cyber Security (ACCS).

### Included Dataset Files

* `UNSW_NB15_training-set.csv`
* `UNSW_NB15_testing-set.csv`
* `NUSW-NB15_features.csv`
* `UNSW-NB15_LIST_EVENTS.csv`

The dataset contains modern network traffic records with both normal and malicious activities.

---

# ⚙️ Working Process

## 1️⃣ Data Collection

The UNSW-NB15 dataset is loaded and merged for training and testing.

## 2️⃣ Data Preprocessing

The system performs:

* Missing value handling
* Label encoding
* Feature selection
* Data normalization

## 3️⃣ Model Training

Machine Learning models are trained on labeled network traffic data.

## 4️⃣ Attack Prediction

The trained model predicts whether incoming traffic is:

* Normal
* Malicious

## 5️⃣ Threat Classification

Detected attacks are categorized into specific attack types.

## 6️⃣ Result Visualization

Outputs are displayed using graphs, confusion matrices, and prediction summaries.

---

# 📈 Machine Learning Workflow

```text
UNSW-NB15 Dataset
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
Model Training
        ↓
Threat Prediction
        ↓
Attack Classification
        ↓
Visualization & Reporting
```

---

# 🧪 Model Evaluation

The trained model is evaluated using standard Machine Learning metrics:

| Metric           | Description                    |
| ---------------- | ------------------------------ |
| Accuracy         | Overall prediction correctness |
| Precision        | Correct positive predictions   |
| Recall           | Detection capability           |
| F1-Score         | Balanced performance measure   |
| Confusion Matrix | Detailed prediction analysis   |

---

# 📌 Use Cases

* Intrusion Detection Systems
* Security Operation Centers (SOC)
* Threat Monitoring
* Cybersecurity Research
* AI-Based Security Systems
* Academic Final Year Projects
* Network Traffic Analysis

---

# 🔒 Future Enhancements

* Real-Time Packet Sniffing
* Deep Learning Integration
* Web-Based Dashboard
* Live Threat Monitoring
* SIEM Integration
* Cloud Deployment
* Automated Threat Response

---

# ▶️ How to Run the Project

```bash
# Clone Repository
git clone <repository-link>

# Navigate to Project Folder
cd IDS-ML-Project

# Install Dependencies
pip install -r requirements.txt

# Run the Project
python main.py
```

---

# 📁 Project Structure

```text
IDS-ML-Project/
│
├── dataset/
├── models/
├── outputs/
├── visuals/
├── main.py
├── train_model.py
├── predict.py
├── requirements.txt
└── README.md
```

---

# 📚 Learning Outcomes

This project demonstrates:

* Practical Machine Learning in Cybersecurity
* Network Threat Detection
* AI-Based Intrusion Analysis
* Dataset Preprocessing Techniques
* Real-Time Prediction Systems
* Security Analytics

---

# 👨‍💻 Author

Developed as a Machine Learning and Cybersecurity project for advanced intrusion detection and intelligent threat analysis.

---

# ⭐ Conclusion

The **Machine Learning Based Intrusion Detection System** demonstrates how Artificial Intelligence can be leveraged to strengthen cybersecurity infrastructures by automatically detecting and classifying malicious network activities.

This project combines Machine Learning, Data Analytics, and Cybersecurity concepts into a powerful intelligent security solution capable of improving modern network defense systems.
