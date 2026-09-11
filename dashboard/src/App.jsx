import { useState, useEffect } from 'react'
import axios from 'axios'
import './App.css'

const API_URL = 'http://127.0.0.1:8000'

function App() {
  const [prediction, setPrediction] = useState(null)
  const [drift, setDrift] = useState(null)
  const [shap, setShap] = useState(null)
  const [loading, setLoading] = useState(false)
  const [apiStatus, setApiStatus] = useState('checking...')
  const [retrainResult, setRetrainResult] = useState(null)
  const [dataCount, setDataCount] = useState(0)

  // Check API status on load
  useEffect(() => {
    checkApiStatus()
  }, [])

  const checkApiStatus = async () => {
    try {
      await axios.get(`${API_URL}/`)
      setApiStatus('✅ Connected')
    } catch (err) {
      setApiStatus('❌ Disconnected')
    }
  }

  // Make a prediction
  const makePrediction = async () => {
    setLoading(true)
    try {
      const res = await axios.get(`${API_URL}/predict`, {
        params: {
          sepal_length: 5.1,
          sepal_width: 3.5,
          petal_length: 1.4,
          petal_width: 0.2
        }
      })
      setPrediction(res.data)
    } catch (err) {
      console.error(err)
    }
    setLoading(false)
  }

  // Check drift
  const checkDrift = async () => {
    try {
      const res = await axios.get(`${API_URL}/drift-report`)
      setDrift(res.data)
    } catch (err) {
      console.error(err)
    }
  }

  // Get SHAP explanation
  const getExplanation = async () => {
    try {
      const res = await axios.get(`${API_URL}/explain`, {
        params: {
          sepal_length: 5.1,
          sepal_width: 3.5,
          petal_length: 1.4,
          petal_width: 0.2
        }
      })
      setShap(res.data)
    } catch (err) {
      console.error(err)
    }
  }

  // Add extreme data to simulate drift
  const addExtremeData = async () => {
    try {
      for (let i = 0; i < 5; i++) {
        const sl = 8 + Math.random() * 2
        const sw = 5 + Math.random() * 1
        const pl = 7 + Math.random() * 1
        const pw = 4 + Math.random() * 1

        await axios.post(`${API_URL}/add-data`, null, {
          params: {
            sepal_length: sl,
            sepal_width: sw,
            petal_length: pl,
            petal_width: pw
          }
        })
      }
      setDataCount(prev => prev + 5)
      alert('Added 5 extreme data points! Now check drift.')
    } catch (err) {
      console.error(err)
      alert('Failed to add data')
    }
  }

  // Trigger auto-retrain
  const triggerRetrain = async () => {
    try {
      const res = await axios.post(`${API_URL}/auto-retrain`)
      setRetrainResult(res.data)
    } catch (err) {
      console.error(err)
      alert('Retrain failed')
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>🌸 Self-Healing ML Pipeline</h1>
        <p className="status">API Status: {apiStatus}</p>
      </header>

      <main className="grid">
        {/* Predict Card */}
        <div className="card">
          <h2>🔮 Prediction</h2>
          <button onClick={makePrediction} disabled={loading}>
            {loading ? 'Predicting...' : 'Make Prediction'}
          </button>
          {prediction && (
            <div className="result">
              <p><strong>Flower:</strong> {prediction.flower}</p>
              <p><strong>Class:</strong> {prediction.prediction}</p>
            </div>
          )}
        </div>

        {/* Drift Card */}
        <div className="card">
          <h2>📊 Drift Monitor</h2>
          <button onClick={checkDrift}>Check Drift</button>
          {drift && (
            <div className="result">
              <p><strong>PSI:</strong> {drift.psi.toFixed(4)}</p>
              <p><strong>Threshold:</strong> {drift.threshold}</p>
              <p className={drift.drift_detected ? 'alert' : 'ok'}>
                {drift.status}
              </p>
            </div>
          )}
          <button 
            onClick={addExtremeData}
            style={{marginTop: '0.5rem', background: '#f59e0b'}}
          >
            ➕ Add Extreme Data (5)
          </button>
          <button 
            onClick={triggerRetrain}
            style={{marginTop: '0.5rem', background: '#dc2626'}}
          >
            🚨 Trigger Auto-Retrain
          </button>
        </div>

        {/* SHAP Card */}
        <div className="card">
          <h2>🧠 SHAP Explanation</h2>
          <button onClick={getExplanation}>Explain Prediction</button>
          {shap && (
            <div className="result">
              {Object.entries(shap.shap_values || {}).map(([key, val]) => (
                <p key={key}>
                  <strong>{key}:</strong> {val}
                </p>
              ))}
            </div>
          )}
        </div>
      </main>

      {/* Retrain Result Modal */}
      {retrainResult && (
        <div className="modal">
          <div className="modal-content">
            <h3>🚨 Auto-Retrain Result</h3>
            <p><strong>Message:</strong> {retrainResult.message}</p>
            <p><strong>Action:</strong> {retrainResult.action}</p>
            {retrainResult.retrain_output && (
              <pre>{retrainResult.retrain_output}</pre>
            )}
            <button onClick={() => setRetrainResult(null)}>Close</button>
          </div>
        </div>
      )}
    </div>
  )
}

export default App