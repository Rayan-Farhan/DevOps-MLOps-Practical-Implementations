import React, { useState } from "react";
import axios from "axios";
import { API_URL } from "../config";

const PredictionForm = ({ setResult }) => {
  const [formData, setFormData] = useState({
    Pregnancies: "",
    Glucose: "",
    BloodPressure: "",
    SkinThickness: "",
    Insulin: "",
    BMI: "",
    DiabetesPedigreeFunction: "",
    Age: ""
  });

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const payload = {
        Pregnancies: parseInt(formData.Pregnancies) || 0,
        Glucose: parseInt(formData.Glucose) || 0,
        BloodPressure: parseInt(formData.BloodPressure) || 0,
        SkinThickness: parseInt(formData.SkinThickness) || 0,
        Insulin: parseInt(formData.Insulin) || 0,
        BMI: parseFloat(formData.BMI) || 0,
        DiabetesPedigreeFunction: parseFloat(formData.DiabetesPedigreeFunction) || 0,
        Age: parseInt(formData.Age) || 0,
      };
      
      const response = await axios.post(`${API_URL}${API_URL.endsWith('/') ? '' : '/'}api/predict`, payload, {
        headers: {
          'Content-Type': 'application/json',
        },
      });
      setResult(response.data);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage = error.response?.data?.detail || error.message || 'Unknown error';
      alert(`Error connecting to backend: ${errorMessage}\n\nCheck browser console (F12) for details.`);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      {Object.keys(formData).map((key) => (
        <div key={key}>
          <label>{key}</label>
          <input
            type="number"
            name={key}
            value={formData[key]}
            onChange={handleChange}
            required
          />
        </div>
      ))}
      <button type="submit">Predict</button>
    </form>
  );
};

export default PredictionForm;