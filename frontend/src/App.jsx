import { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [formData, setFormData] = useState({
    gender: "Female",
    SeniorCitizen: 0,
    Partner: "No",
    Dependents: "No",
    tenure: 5,
    PhoneService: "Yes",
    MultipleLines: "No",
    InternetService: "Fiber optic",
    OnlineSecurity: "No",
    OnlineBackup: "No",
    DeviceProtection: "No",
    TechSupport: "No",
    StreamingTV: "Yes",
    StreamingMovies: "Yes",
    Contract: "Month-to-month",
    PaperlessBilling: "Yes",
    PaymentMethod: "Electronic check",
    MonthlyCharges: 89.1,
    TotalCharges: 420.5,
  });

  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"].includes(name)
        ? Number(value)
        : value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post("http://127.0.0.1:8000/predict", formData);
      setResult(response.data);
    } catch (error) {
      console.error(error);
      alert("Prediction failed");
    }
  };

  return (
    <div style={{ maxWidth: "900px", margin: "0 auto", padding: "20px" }}>
      <h1>Telecom Customer Churn Prediction</h1>

      <form onSubmit={handleSubmit} style={{ display: "grid", gap: "12px" }}>
        {Object.keys(formData).map((key) => (
          <div key={key}>
            <label>{key}</label>
            <input
              name={key}
              value={formData[key]}
              onChange={handleChange}
              style={{ width: "100%", padding: "8px" }}
            />
          </div>
        ))}
        <button type="submit">Predict</button>
      </form>

      {result && (
        <div style={{ marginTop: "20px", padding: "16px", border: "1px solid #ccc" }}>
          <h2>Prediction Result</h2>
          <p><strong>Prediction:</strong> {result.prediction}</p>
          <p><strong>Churn Probability:</strong> {result.churn_probability}</p>
          <p><strong>Risk Level:</strong> {result.risk_level}</p>
        </div>
      )}
    </div>
  );
}

export default App;