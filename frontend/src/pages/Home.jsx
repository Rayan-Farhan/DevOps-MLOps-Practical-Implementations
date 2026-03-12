import React, { useState, useEffect } from "react";
import PredictionForm from "../components/PredictionForm";
import ResultCard from "../components/ResultCard";
import axios from "axios";
import { API_URL } from "../config";

export const Home = () => {
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState("checking");

  useEffect(() => {
    const healthCheckUrl = `${API_URL}${API_URL.endsWith("/") ? "" : "/"}api/health`;
    axios
      .get(healthCheckUrl)
      .then(() => setBackendStatus("connected"))
      .catch((err) => {
        console.error("Backend connection failed:", err);
        setBackendStatus("disconnected");
      });
  }, []);

  const statusClass =
    backendStatus === "connected"
      ? "status-connected"
      : backendStatus === "disconnected"
        ? "status-disconnected"
        : "status-checking";

  const statusText =
    backendStatus === "checking"
      ? "Checking connection..."
      : backendStatus === "connected"
        ? "Backend connected"
        : "Backend disconnected";

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Diabetes Risk Predictor</h1>
        <p className="app-subtitle">
          Enter patient health parameters to predict diabetes risk using an ML model.
        </p>
      </header>

      <div className={`status-banner ${statusClass}`}>
        <span className="status-dot" />
        {statusText}
      </div>

      {error && (
        <div className="error-banner">
          <strong>Error:</strong> {error}
        </div>
      )}

      <PredictionForm setResult={setResult} setError={setError} />

      {result && <ResultCard result={result} />}
    </div>
  );
};
