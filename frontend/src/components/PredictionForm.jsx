import React, { useState } from "react";
import axios from "axios";
import { API_URL } from "../config";

const FIELDS = [
  { key: "Pregnancies",              label: "Pregnancies",             min: 0, max: 20,   step: 1,    type: "int",   hint: "0 - 20" },
  { key: "Glucose",                  label: "Glucose (mg/dL)",         min: 0, max: 300,  step: 1,    type: "int",   hint: "0 - 300" },
  { key: "BloodPressure",            label: "Blood Pressure (mm Hg)",  min: 0, max: 200,  step: 1,    type: "int",   hint: "0 - 200" },
  { key: "SkinThickness",            label: "Skin Thickness (mm)",     min: 0, max: 100,  step: 1,    type: "int",   hint: "0 - 100" },
  { key: "Insulin",                  label: "Insulin (mu U/mL)",       min: 0, max: 900,  step: 1,    type: "int",   hint: "0 - 900" },
  { key: "BMI",                      label: "BMI (kg/m\u00B2)",        min: 0, max: 80,   step: 0.1,  type: "float", hint: "0 - 80" },
  { key: "DiabetesPedigreeFunction", label: "Diabetes Pedigree Fn",    min: 0, max: 3,    step: 0.01, type: "float", hint: "0 - 3.0" },
  { key: "Age",                      label: "Age (years)",             min: 0, max: 120,  step: 1,    type: "int",   hint: "0 - 120" },
];

const initialFormData = Object.fromEntries(FIELDS.map((f) => [f.key, ""]));

const PredictionForm = ({ setResult, setError }) => {
  const [formData, setFormData] = useState(initialFormData);
  const [fieldErrors, setFieldErrors] = useState({});
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
    // Clear field error on change
    if (fieldErrors[name]) {
      setFieldErrors((prev) => ({ ...prev, [name]: null }));
    }
  };

  const validate = () => {
    const errors = {};
    for (const f of FIELDS) {
      const raw = formData[f.key];
      if (raw === "" || raw === undefined) {
        errors[f.key] = "Required";
        continue;
      }
      const val = f.type === "int" ? parseInt(raw, 10) : parseFloat(raw);
      if (isNaN(val))        errors[f.key] = "Must be a number";
      else if (val < f.min)  errors[f.key] = `Min value is ${f.min}`;
      else if (val > f.max)  errors[f.key] = `Max value is ${f.max}`;
    }
    return errors;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setResult(null);

    const errors = validate();
    setFieldErrors(errors);
    if (Object.keys(errors).length > 0) return;

    const payload = {};
    for (const f of FIELDS) {
      payload[f.key] = f.type === "int"
        ? parseInt(formData[f.key], 10)
        : parseFloat(formData[f.key]);
    }

    setLoading(true);
    try {
      const url = `${API_URL}${API_URL.endsWith("/") ? "" : "/"}api/predict`;
      const response = await axios.post(url, payload, {
        headers: { "Content-Type": "application/json" },
      });
      setResult(response.data);
    } catch (error) {
      console.error("Prediction error:", error);
      const detail = error.response?.data?.detail;
      if (Array.isArray(detail)) {
        setError(detail.map((d) => d.msg || JSON.stringify(d)).join("; "));
      } else {
        setError(detail || error.message || "An unexpected error occurred");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFormData(initialFormData);
    setFieldErrors({});
    setError(null);
    setResult(null);
  };

  return (
    <form className="prediction-form" onSubmit={handleSubmit} noValidate>
      <div className="form-grid">
        {FIELDS.map((f) => (
          <div key={f.key} className={`form-field ${fieldErrors[f.key] ? "has-error" : ""}`}>
            <label htmlFor={f.key}>{f.label}</label>
            <input
              id={f.key}
              type="number"
              name={f.key}
              value={formData[f.key]}
              onChange={handleChange}
              placeholder={f.hint}
              min={f.min}
              max={f.max}
              step={f.step}
            />
            {fieldErrors[f.key] ? (
              <span className="field-error">{fieldErrors[f.key]}</span>
            ) : (
              <span className="field-hint">Range: {f.hint}</span>
            )}
          </div>
        ))}
      </div>
      <div className="form-actions">
        <button type="submit" className="btn btn-primary" disabled={loading}>
          {loading ? "Predicting..." : "Predict"}
        </button>
        <button type="button" className="btn btn-secondary" onClick={handleReset}>
          Reset
        </button>
      </div>
    </form>
  );
};

export default PredictionForm;
