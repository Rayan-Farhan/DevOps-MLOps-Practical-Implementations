import React, { useState, useEffect } from "react";
import PredictionForm from "../components/PredictionForm";
import ResultCard from "../components/ResultCard";
import axios from "axios";
import { API_URL } from "../config";

export const Home = () => {
  const [result, setResult] = useState(null);
  const [backendStatus, setBackendStatus] = useState("checking");

  useEffect(() => {
    const healthCheckUrl = `${API_URL}${API_URL.endsWith('/') ? '' : '/'}api/health`;
    axios.get(healthCheckUrl)
      .then((response) => {
        setBackendStatus("connected");
      })
      .catch((error) => {
        console.error("Backend connection failed:", error);
        setBackendStatus("disconnected");
      });
  }, []);

  return (
    <div className="app-container">
      <h1>Diabetes Prediction</h1>
      
      <div style={{ 
        padding: "10px", 
        marginBottom: "20px", 
        borderRadius: "5px",
        backgroundColor: backendStatus === "connected" ? "#d4edda" : backendStatus === "disconnected" ? "#f8d7da" : "#fff3cd",
        color: backendStatus === "connected" ? "#155724" : backendStatus === "disconnected" ? "#721c24" : "#856404",
        border: `1px solid ${backendStatus === "connected" ? "#c3e6cb" : backendStatus === "disconnected" ? "#f5c6cb" : "#ffeaa7"}`
      }}>
        Backend Status: {
          backendStatus === "checking" ? "🔄 Checking..." :
          backendStatus === "connected" ? "✅ Connected" :
          "❌ Disconnected - Check console (F12) for details"
        }
      </div>

      <PredictionForm setResult={setResult} />
      {result && <ResultCard result={result} />}
    </div>
  );
};