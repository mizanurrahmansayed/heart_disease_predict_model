# ❤️ Heart Disease Prediction Web App

🚀 **Live Demo:**  
👉 https://huggingface.co/spaces/Sayed275262/heart_disease_predictor 

---

## 📋 Project Description

### 🎯 Core Value Proposition

This project is an advanced heart disease prediction web application built with Gradient Boosting algorithms, achieving high-precision predictions (87.50% accuracy). Through an intuitive Gradio interface, medical professionals and general users can quickly assess patients' cardiovascular risk, providing data-driven support for early prevention and timely intervention.

### 💡 Project Mission

Utilize artificial intelligence technology to democratize heart disease risk assessment, making high-quality medical prediction tools easily accessible and usable, contributing to global cardiovascular disease prevention efforts.

## ✨ Key Features

### 🔬 Machine Learning Capabilities
- ⚡ High-Precision Prediction: 87.50% accuracy using Gradient Boosting  
- 🧠 Intelligent Feature Engineering: BP/HR ratio, cholesterol/age ratio  
- 📊 Probabilistic Output: Disease & no-disease probability  

### 🖥️ User Experience
- 🎨 Modern UI using Gradio Soft theme  
- 📱 Responsive (desktop & mobile)  
- 🎛️ Interactive controls (slider, dropdown, radio)  
- 💡 Built-in sample test cases  

### 🛠️ Technical Highlights
- 🐍 Python ML ecosystem  
- 🔄 Real-time prediction  
- 🌐 Web deployment with public sharing  
- 🔧 Proxy optimization (Windows fix)  

### 📈 Data Science
- 🎯 15-dimensional feature analysis  
- 🔍 Multi-factor health assessment  
- 📋 Medical-standard encoding  

---
## 🖼️ Demo Preview
![App Screenshot](your-image-link)

---

## 🚀 Installation Guide

### 📋 System Requirements
- OS: Windows / macOS / Linux  
- Python: 3.8+  
- RAM: Minimum 4GB (8GB recommended)  
- Internet: Required for setup  

---

### 🔧 Quick Start


### 1️⃣ Clone Project
git clone https://github.com/mizanurrahmansayed/heart_disease_predict_model.git
cd heart_disease_model

### 2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows

### macOS/Linux
source venv/bin/activate

### 3️⃣ Install Dependencies
pip install -r requirements.txt

### 4️⃣ Verify Installation
python -c "import gradio, sklearn, pandas, numpy; print('✅ Installed!')"
### 📦 Core Dependencies
- gradio==6.3.0 - Web interface framework
- scikit-learn==1.3.2 - Machine learning library
- pandas==2.3.3 - Data processing
- numpy==2.4.1 - Numerical computing
- joblib==1.5.3 - Model serialization
### 💻 Usage Instructions
### 🎮 Launch Application
#### Method 1: Direct Run (Recommended)
- python app.py
#### Method 2: Run with Custom Port
- python app.py --port 8080  # Modify port number
### 🌐 Access Application
- Local Access: http://127.0.0.1:7860
- Network Access: http://localhost:7860
- Public Sharing: Temporary public link generated on startup (share=True)
📝 Usage Examples
Example 1: Typical Middle-aged Male Patient
## 📝 Usage Examples

### 👨 Example 1: Typical Patient
#### Enter the following values:

### 👨 Example 1: Typical Patient

| Parameter        | Value |
|-----------------|------|
| Age             | 40 |
| Sex             | Male |
| ChestPainType   | ATA |
| RestingBP       | 130 mm Hg |
| Cholesterol     | 250 mg/dl |
| FastingBS       | 0 |
| RestingECG      | Normal |
| MaxHR           | 165 bpm |
| ExerciseAngina  | No |
| Oldpeak         | 1.0 |
| ST_Slope        | Up |

---
### ⚠️ Example 2: High-risk Female Patient


| Parameter        | Value |
|-----------------|------|
| Age             | 50 |
| Sex             | Female |
| ChestPainType   | ASY (Asymptomatic) |
| RestingBP       | 140 mm Hg |
| Cholesterol     | 300 mg/dl |
| FastingBS       | 1 (> 120 mg/dl) |
| RestingECG      | LVH (Left Ventricular Hypertrophy) |
| MaxHR           | 140 bpm |
| ExerciseAngina  | Yes |
| Oldpeak         | 2.5 mm |
| ST_Slope        | Flat |

---
### ✅ Example 3: Low-risk Young Patient

| Parameter        | Value |
|-----------------|------|
| Age             | 25 |
| Sex             | Male |
| ChestPainType   | NAP (Non-Anginal Pain) |
| RestingBP       | 120 mm Hg |
| Cholesterol     | 180 mg/dl |
| FastingBS       | 0 |
| RestingECG      | Normal |
| MaxHR           | 180 bpm |
| ExerciseAngina  | No |
| Oldpeak         | 0.0 mm |
| ST_Slope        | Up |

## 📊 Interpreting Prediction Results

json
{
  "Prediction": "💚 No Heart Disease",
  "Confidence": "86.34%",
  "Probability of Disease": "13.66%",
  "Probability of No Disease": "86.34%"
}
### ⚙️ Configuration
| Parameter   | Default   | Description    |
| ----------- | --------- | -------------- |
| server_name | 127.0.0.1 | Server address |
| server_port | 7860      | App port       |
| share       | True      | Public link    |
| debug       | True      | Debug mode     |

### 🔧 Environment Variables
Application automatically sets the following environment variables to avoid proxy issues:

- NO_PROXY=localhost,127.0.0.1,::1,0.0.0.0
- HTTP_PROXY=(cleared)
- HTTPS_PROXY=(cleared)
### 🎛️ Application Parameters
In app.py, the demo.launch() function accepts these configurable parameters:

### Parameter	Default	Description
- server_name	127.0.0.1	Server address
- server_port	7860	Service port
- share	True	Generate public link
- debug	True	Debug mode
- theme	gr.themes.Soft()	UI theme
### 📁 File Path Configuration
- Model File: heart_disease_model.pkl (must be in same directory as app.py)
- Log Output: Real-time console display
- Temporary Files: System temp directory
### 📚 API Reference
### 🔌 Core Function Interfaces
- load_model(model_path='heart_disease_model.pkl')
- Loads the pre-trained cardiovascular disease prediction model.

Parameters:

- model_path (str): Model file path, defaults to 'heart_disease_model.pkl'
Returns:

Deserialized scikit-learn model object
Exceptions:
- FileNotFoundError: Model file not found
- RuntimeError: Model loading failed
- predict_heart_disease(...)
- Core function for performing heart disease risk prediction.

## 📚 API Reference

### 🔌 `predict_heart_disease()`

Performs heart disease risk prediction based on patient medical data.

---

## 📥 Input Parameters

| Parameter        | Type   | Range / Options                  | Description |
|-----------------|--------|----------------------------------|------------|
| Age             | float  | 10 – 110                        | Patient age (years) |
| Sex             | str    | Female, Male                    | Gender |
| ChestPainType   | str    | ASY, ATA, NAP, TA               | Chest pain type |
| RestingBP       | float  | 80 – 160                        | Resting blood pressure (mmHg) |
| Cholesterol     | float  | 40 – 400                        | Serum cholesterol (mg/dl) |
| FastingBS       | int    | 0, 1                            | Fasting blood sugar > 120 mg/dl (1 = Yes) |
| RestingECG      | str    | LVH, Normal, ST                 | ECG results |
| MaxHR           | float  | 60 – 220                        | Maximum heart rate (bpm) |
| ExerciseAngina  | str    | No, Yes                         | Exercise-induced angina |
| Oldpeak         | float  | 0.0 – 6.0                       | ST depression |
| ST_Slope        | str    | Down, Flat, Up                  | ST segment slope |

---

## 📤 Return Value

json
{
  "Prediction": "Heart Disease Detected", 
  
  "Confidence": "85.32%",
  
  "Probability of Disease": "85.32%",
  
  "Probability of No Disease": "14.68%"
}

### 👥 Contributing Guidelines
### 🤝 How to Contribute
We welcome all forms of contributions! Please follow these steps:

### 1️⃣ Fork & Clone
- git clone https://github.com/mizanurrahmansayed/heart_disease_predict_model.git
- cd heart_disease_model
- git checkout -b feature/your-feature-name
### 2️⃣ Development Standards
-🐍 Code Style: Follow PEP 8 Python code standards
-📝 Documentation: All functions must include docstrings
-🧪 Test Coverage: New features require unit tests
-🔍 Code Review: Self-review before submission
### 3️⃣ Commit Standards
#### Commit message format: <type>(<scope>): <subject>
- git commit -m "feat(model): Add new feature engineering method"
- git commit -m "fix(ui): Fix mobile display issues"
- git commit -m "docs(readme): Update installation instructions"
Commit Types:

feat: New feature
fix: Bug fix
docs: Documentation updates
style: Code formatting adjustments
refactor: Refactoring
test: Testing related
chore: Build process or auxiliary tool changes
### 4️⃣ Pull Request
📋 Detailed description of changes and motivation
### 🔗 Link related Issues (if any)
- ✅ Ensure all tests pass
### 📸 Provide screenshots or demos if applicable
- 🐛 Bug Reports
Please use GitHub Issues template including:

### 🖥️ Operating system and environment information
- 📋 Reproduction steps
- 📷 Error screenshots or logs
- 🔍 Expected vs actual behavior
- 💡 Feature Suggestions
We particularly welcome suggestions for:

- 🎨 UI/UX improvements
- 🤖 New algorithm integration
- 📊 Visualization enhancements
- 🔧 Performance optimizations
- 🌐 Multi-language support

### 📞 Contact Information
👤 Project Maintainer
- Name: AI/ML Expert Team
- Organization: AI/ML WEB TESTING
- Email: ai.ml.expert@example.com
- GitHub: @ai-ml-expert
### 🌟 Technical Support
📚 Documentation: Check this README and code comments
- 🐛 Bug Reports: GitHub Issues
- 💬 Discussions: GitHub Discussions
- 📧 Email: ai.ml.expert@example.com
### 🤝 Collaboration Opportunities
If you are:

- 🏥 Healthcare Institutions: Seeking customized deployment solutions
- 🔬 Researchers: Wanting to collaborate on model improvements
- 💼 Enterprise Clients: Needing commercial solutions
- 🎓 Educational Institutions: Using for teaching and research purposes
Welcome to contact us for deep collaboration!

### 🙏 Acknowledgments
- 📚 Technology Stack Thanks
- 🤖 Scikit-Learn Team: Providing excellent machine learning framework
- 🎨 Gradio Team: Creating simple yet powerful web ML interfaces
- 🐍 Python Community: Continuously advancing open-source AI development
- 📊 Medical Dataset Providers: Providing valuable data for model training
#### 🌟 Special Thanks
Thanks to all researchers, developers, and healthcare workers contributing to cardiovascular health!
