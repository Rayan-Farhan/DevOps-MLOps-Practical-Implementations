import React from "react";

const ResultCard = ({ result }) => {
  const isDiabetic = result.prediction === 1;

  return (
    <div className={`result-card ${isDiabetic ? "result-diabetic" : "result-healthy"}`}>
      <div className="result-icon">{isDiabetic ? "\u26A0" : "\u2714"}</div>
      <h2 className="result-title">
        {isDiabetic ? "Diabetic" : "Non-Diabetic"}
      </h2>
      <p className="result-description">
        {isDiabetic
          ? "The model predicts the patient is likely diabetic. Please consult a healthcare professional for clinical evaluation."
          : "The model predicts the patient is unlikely to have diabetes based on the provided parameters."}
      </p>
    </div>
  );
};

export default ResultCard;
