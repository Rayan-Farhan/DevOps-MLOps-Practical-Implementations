from pydantic import BaseModel, Field

class DiabetesInput(BaseModel):
    Pregnancies: int = Field(..., ge=0, le=20, example=2, description="Number of pregnancies (0-20)")
    Glucose: int = Field(..., ge=0, le=300, example=120, description="Plasma glucose concentration (0-300)")
    BloodPressure: int = Field(..., ge=0, le=200, example=70, description="Diastolic blood pressure (mm Hg, 0-200)")
    SkinThickness: int = Field(..., ge=0, le=100, example=20, description="Triceps skinfold thickness (mm, 0-100)")
    Insulin: int = Field(..., ge=0, le=900, example=85, description="2-Hour serum insulin (mu U/ml, 0-900)")
    BMI: float = Field(..., ge=0.0, le=80.0, example=28.5, description="Body mass index (0-80)")
    DiabetesPedigreeFunction: float = Field(..., ge=0.0, le=3.0, example=0.45, description="Diabetes pedigree function (0-3)")
    Age: int = Field(..., ge=0, le=120, example=33, description="Age in years (0-120)")

    class Config:
        schema_extra = {
            "example": {
                "Pregnancies": 2,
                "Glucose": 120,
                "BloodPressure": 70,
                "SkinThickness": 20,
                "Insulin": 85,
                "BMI": 28.5,
                "DiabetesPedigreeFunction": 0.45,
                "Age": 33
            }
        }