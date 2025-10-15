import React, { useState } from "react";
import axios from "axios";

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
      const response = await axios.post("http://127.0.0.1:8000/api/predict", formData);
      setResult(response.data);
    } catch (error) {
      console.error(error);
      alert("Error connecting to backend");
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