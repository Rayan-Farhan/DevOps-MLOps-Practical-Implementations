from pydantic import BaseModel, Field

class DiabetesInput(BaseModel):
    Pregnancies: int = Field(..., example=2)
    Glucose: int = Field(..., example=120)
    BloodPressure: int = Field(..., example=70)
    SkinThickness: int = Field(..., example=20)
    Insulin: int = Field(..., example=85)
    BMI: float = Field(..., example=28.5)
    DiabetesPedigreeFunction: float = Field(..., example=0.45)
    Age: int = Field(..., example=33)

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