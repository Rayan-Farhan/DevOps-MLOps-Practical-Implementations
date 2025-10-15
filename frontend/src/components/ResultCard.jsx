import React from "react";

const ResultCard = ({ result }) => {
  return (
    <div>
      <h2>Prediction Result</h2>
      <p>
        {result.prediction === 1
          ? "⚠️ The model predicts the patient is Diabetic..."
          : "✅ The model predicts the patient is Not Diabetic!!!"}
      </p>
    </div>
  );
};

export default ResultCard;