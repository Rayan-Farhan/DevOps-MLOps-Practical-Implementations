import React, { useState } from "react";
import PredictionForm from "../components/PredictionForm";
import ResultCard from "../components/ResultCard";

export const Home = () => {
  const [result, setResult] = useState(null);

  return (
    <div className="app-container">
      <h1>Diabetes Prediction</h1>
      <PredictionForm setResult={setResult} />
      {result && <ResultCard result={result} />}
    </div>
  );
};