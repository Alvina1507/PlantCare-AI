# 🌿 PlantCare AI – Intelligent Plant Disease Detection System

PlantCare AI is an AI-powered plant disease screening application that uses deep learning to identify likely plant diseases from leaf images.

Users can upload a plant leaf image and receive a predicted disease category, confidence score and relevant plant-care information.

## ✨ Features

- AI-based plant disease detection
- Image upload and preview
- Top 3 prediction results
- Prediction confidence scores
- Plant disease guide
- Symptoms, causes and prevention information
- Interactive dashboard with a dark green UI

## 🧠 AI Model

- **Architecture:** MobileNetV2
- **Technique:** Transfer Learning
- **Dataset:** PlantVillage
- **Image Size:** 224 × 224
- **Number of Classes:** 39
- **Validation Accuracy:** 96.27%

The model was trained using TensorFlow and Keras.

*Note: Validation accuracy is measured on the held-out PlantVillage validation split. Real-world performance may differ.*

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- Streamlit
- NumPy
- Pillow

## 📁 Project Structure

```text
PlantCareAI/
│
├── app.py
├── train_model.py
├── dataset.py
├── check.py
├── plant_disease_model.keras
├── best_plant_model.keras
├── class_names.json
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Alvina1507/PlantCare-AI.git
cd PlantCare-AI
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📊 Model Performance

| Metric | Result |
|---|---:|
| Model | MobileNetV2 |
| Classes | 39 |
| Validation Accuracy | 96.27% |
| Validation Loss | 0.1154 |

## 📚 Dataset

This project uses the PlantVillage dataset for training and validation.

The dataset is not included in this repository. Obtain the dataset separately if you want to retrain the model.

## ⚠️ Disclaimer

PlantCare AI is an educational plant disease screening tool. Its predictions are not a confirmed agricultural diagnosis. For important crop-health decisions, consult an agricultural expert.

## 👩‍💻 Developer

**Alvina S**

B.Tech – Computer Science and Engineering  
Jain (Deemed-to-be) University, Bengaluru

## 🔗 GitHub Repository

https://github.com/Alvina1507/PlantCare-AI
