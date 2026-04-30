import os
import gradio as gr
import numpy as np
import pickle
import pandas as pd


os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
# Most effective: tell requests / urllib not to use proxy for localhost
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1,0.0.0.0"

# Load your trained model
def load_model(model_path='heart_disease_model.pkl'):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

model = load_model('heart_disease_model.pkl')

# Feature names (exact order!)
feature_names = [
    'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
    'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak',
    'ST_Slope', 'BP_HR_Ratio', 'Chol_Age_Ratio', 'Risk_Score', 'Cardiac_Stress'
]

# Mappings (ensure they match your training script exactly)
sex_mapping = {'Female': 0, 'Male': 1}
chest_pain_mapping = {'ASY': 0, 'ATA': 1, 'NAP': 2, 'TA': 3}
resting_ecg_mapping = {'LVH': 0, 'Normal': 1, 'ST': 2}
exercise_angina_mapping = {'No': 0, 'Yes': 1}
st_slope_mapping = {'Down': 0, 'Flat': 1, 'Up': 2}

def predict_heart_disease(
    Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS,
    RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope
):
    try:
        # ── Type conversions ──
        Age = float(Age)
        RestingBP = float(RestingBP)
        Cholesterol = float(Cholesterol)
        MaxHR = float(MaxHR)
        Oldpeak = float(Oldpeak)
        FastingBS = int(FastingBS)

        # ── Encoding ──
        Sex_enc = sex_mapping[Sex]
        ChestPainType_enc = chest_pain_mapping[ChestPainType]
        RestingECG_enc = resting_ecg_mapping[RestingECG]
        ExerciseAngina_enc = exercise_angina_mapping[ExerciseAngina]
        ST_Slope_enc = st_slope_mapping[ST_Slope]

        # ── Engineered features ──
        BP_HR_Ratio    = RestingBP / (MaxHR + 1)
        Chol_Age_Ratio = Cholesterol / (Age + 1)
        Risk_Score     = (Oldpeak * 2) + (Age / 10)
        Cardiac_Stress = MaxHR / (RestingBP + 1)

        # ── Create dictionary with column names (exact names from training) ──
        features_dict = {
            'Age': Age,
            'Sex': Sex_enc,
            'ChestPainType': ChestPainType_enc,
            'RestingBP': RestingBP,
            'Cholesterol': Cholesterol,
            'FastingBS': FastingBS,
            'RestingECG': RestingECG_enc,
            'MaxHR': MaxHR,
            'ExerciseAngina': ExerciseAngina_enc,
            'Oldpeak': Oldpeak,
            'ST_Slope': ST_Slope_enc,
            'BP_HR_Ratio': BP_HR_Ratio,
            'Chol_Age_Ratio': Chol_Age_Ratio,
            'Risk_Score': Risk_Score,
            'Cardiac_Stress': Cardiac_Stress
        }

        # ── Convert to DataFrame (this is the key fix) ──
        input_df = pd.DataFrame([features_dict])   # shape (1, n_features) with column names

        # ── Predict ──
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        result = "🚨Heart Disease Detected" if prediction == 1 else "💚No Heart Disease"
        confidence = probability[1] if prediction == 1 else probability[0]

        return {
            "Prediction": result,
            "Confidence": f"{confidence:.2%}",
            "Probability of Disease": f"{probability[1]:.2%}",
            "Probability of No Disease": f"{probability[0]:.2%}"
        }

    except KeyError as e:
        return {"Error": f"Invalid input value (check categories): {str(e)}"}
    except Exception as e:
        return {"Error": f"Prediction failed: {str(e)}"}
# ───────────────────────────────────────────────
# Gradio UI (Blocks – recommended over Interface)
# ───────────────────────────────────────────────

with gr.Blocks() as demo:   # ← no theme here anymore
    gr.Markdown("# 🫀🩺 Heart Disease Prediction Model 🩺 ")
    gr.Markdown("Gradient Boosting model (reported accuracy: 87.50%)")

    with gr.Row():
        with gr.Column():
            gr.Markdown("## 🧑‍⚕️Patient Information")
            Age = gr.Slider(10, 110, value=50, step=1, label="Age (years)")
            Sex = gr.Radio(["Female", "Male"], label="Sex", value="Male")
            ChestPainType = gr.Dropdown(
                choices=["ASY", "ATA", "NAP", "TA"], label="Chest Pain Type", value="ATA"
            )
            RestingBP = gr.Slider(80, 160, value=120, step=1, label="Resting Blood Pressure (mm Hg)")
            Cholesterol = gr.Slider(40, 400, value=180, step=1, label="Cholesterol (mg/dl)")
            FastingBS = gr.Radio([0, 1], label="Fasting Blood Sugar > 120 mg/dl (1=Yes)", value=0)

        with gr.Column():
            gr.Markdown("## 📊 Medical Measurements")
            RestingECG = gr.Dropdown(
                choices=["LVH", "Normal", "ST"], label="Resting ECG", value="Normal"
            )
            MaxHR = gr.Slider(60, 220, value=150, step=1, label="Maximum Heart Rate (bpm)")
            ExerciseAngina = gr.Radio(["No", "Yes"], label="Exercise Induced Angina", value="No")
            Oldpeak = gr.Slider(0.0, 6.0, value=1.0, step=0.1, label="Oldpeak (mm)")
            ST_Slope = gr.Dropdown(
                choices=["Down", "Flat", "Up"], label="ST Slope", value="Up"
            )

    predict_btn = gr.Button("❤️Predict Heart Disease", variant="primary")
    output_json = gr.JSON(label="Prediction Results")

    gr.Examples(
        examples=[
            [40, "Male", "ATA", 130, 250, 0, "Normal", 165, "No", 1.0, "Up"],
            [50, "Female", "ASY", 140, 300, 1, "LVH", 140, "Yes", 2.5, "Flat"],
            [25, "Male", "NAP", 120, 180, 0, "Normal", 180, "No", 0.0, "Up"]
        ],
        inputs=[Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS,
                RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope],
        label="Quick Test Examples"
    )

    predict_btn.click(
        fn=predict_heart_disease,
        inputs=[Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS,
                RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope],
        outputs=output_json
    )

# ───────────────────────────────────────────────
# Launch (theme goes here!)
# ───────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Glass(
            primary_hue="red",
            secondary_hue="blue",
            neutral_hue="gray"
        ),
        debug=True
    )
