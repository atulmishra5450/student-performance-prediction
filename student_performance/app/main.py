from pathlib import Path
import pickle

import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates


# ============================================================
# PATHS
# ========================================================
# ====

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "Notebook" / "model.pkl"
TEMPLATES_DIR = BASE_DIR / "app" / "templates"

print("=" * 50)
print("BASE DIR:", BASE_DIR)
print("MODEL PATH:", MODEL_PATH)
print("TEMPLATES DIR:", TEMPLATES_DIR)
print("=" * 50)


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

print("Model loaded successfully!")
print("Model type:", type(model))
print("=" * 50)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Student Performance Prediction",
    description="Predict Mathematics Score",
    version="1.0.0"
)


# ============================================================
# TEMPLATES
# ============================================================

if not TEMPLATES_DIR.exists():
    raise FileNotFoundError(
        f"Templates directory not found: {TEMPLATES_DIR}"
    )

templates = Jinja2Templates(
    directory=str(TEMPLATES_DIR)
)


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "prediction": None,
            "error": None
        }
    )


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
async def predict(
    gender: str = Form(...),
    race_ethnicity: str = Form(...),
    parental_level_of_education: str = Form(...),
    lunch: str = Form(...),
    test_preparation_course: str = Form(...),
    reading_score: float = Form(...),
    writing_score: float = Form(...)
):

    try:

        # ------------------------------------------------------
        # Create input DataFrame
        # ------------------------------------------------------

        input_data = pd.DataFrame({
            "gender": [gender],
            "race_ethnicity": [race_ethnicity],
            "parental_level_of_education": [
                parental_level_of_education
            ],
            "lunch": [lunch],
            "test_preparation_course": [
                test_preparation_course
            ],
            "reading_score": [reading_score],
            "writing_score": [writing_score]
        })

        print("Input received:")
        print(input_data)
        print("=" * 40)

        # ------------------------------------------------------
        # IMPORTANT
        #
        # Your saved model was trained after encoding the
        # categorical columns using get_dummies.
        #
        # Therefore we reproduce the SAME encoding here.
        # ------------------------------------------------------

        categorical_columns = [
            "gender",
            "race_ethnicity",
            "parental_level_of_education",
            "lunch",
            "test_preparation_course"
        ]

        input_data = pd.get_dummies(
            input_data,
            columns=categorical_columns,
            drop_first=False
        )

        # ------------------------------------------------------
        # Get the exact columns used during model training
        # ------------------------------------------------------

        if hasattr(model, "feature_names_in_"):

            expected_columns = list(
                model.feature_names_in_
            )

            # Add missing columns
            for column in expected_columns:

                if column not in input_data.columns:
                    input_data[column] = 0

            # Remove unexpected columns
            input_data = input_data[
                expected_columns
            ]

        print("Final input:")
        print(input_data)
        print("Shape:", input_data.shape)
        print("=" * 40)

        # ------------------------------------------------------
        # Prediction
        # ------------------------------------------------------

        prediction = model.predict(input_data)

        predicted_score = float(prediction[0])

        # Keep score within the normal 0-100 range
        predicted_score = max(
            0,
            min(100, predicted_score)
        )

        print("Predicted Math Score:", predicted_score)
        print("=" * 40)

        return JSONResponse({
            "success": True,
            "prediction": round(
                predicted_score,
                2
            )
        })

    except Exception as e:

        print("PREDICTION ERROR:")
        print(repr(e))
        print("=" * 40)

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )