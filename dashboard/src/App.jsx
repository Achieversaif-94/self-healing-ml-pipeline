import { useState, useEffect } from 'react'
import axios from 'axios'
import './App.css'

const API_URL = 'http://127.0.0.1:8000'

function App() {
  const [prediction, setPrediction] = useState(null)
  const [drift, setDrift] = useState(null)
  const [shap, setShap] = useState(null)
  const [loading, setLoading] = useState(false)
  const [apiStatus, setApiStatus] = useState('checking')
  const [retrainResult, setRetrainResult] = useState(null)
  const [theme, setTheme] = useState(() => {
    // Check localStorage or system preference
    const saved = localStorage.getItem('theme')
    if (saved) return saved
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  })

  // Apply theme to document
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('theme', theme)
  }, [theme])

  useEffect(() => {
    checkApiStatus()
    checkDrift()
  }, [])

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light')
  }

  const checkApiStatus = async () => {
    try {
      await axios.get(`${API_URL}/`)
      setApiStatus('connected')
    } catch {
      setApiStatus('disconnected')
    }
  }

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

  const checkDrift = async () => {
    try {
      const res = await axios.get(`${API_URL}/drift-report`)
      setDrift(res.data)
    } catch (err) {
      console.error(err)
    }
  }

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

  const addExtremeData = async () => {
    try {
      for (let i = 0; i < 5; i++) {
        await axios.post(`${API_URL}/add-data`, null, {
          params: {
            sepal_length: 8 + Math.random() * 2,
            sepal_width: 5 + Math.random() * 1,
            petal_length: 7 + Math.random() * 1,
            petal_width: 4 + Math.random() * 1
          }
        })
      }
      await checkDrift()
    } catch (err) {
      console.error(err)
    }
  }

  const triggerRetrain = async () => {
    try {
      const res = await axios.post(`${API_URL}/auto-retrain`)
      setRetrainResult(res.data)
      await checkDrift()
    } catch (err) {
      console.error(err)
    }
  }

  const driftPercent = drift ? Math.min(drift.psi / drift.threshold, 2) * 50 : 0

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-left">
          <div className="logo">
            <span className="logo-mark">SH</span>
            <span className="logo-text">Self-Healing Pipeline</span>
          </div>
        </div>
        <div className="header-right">
          <div className={`status-pill status-${apiStatus}`}>
            <span className="status-dot" />
            {apiStatus === 'connected' ? 'API Connected' : 
             apiStatus === 'disconnected' ? 'API Offline' : 'Checking...'}
          </div>
          <button 
            className="theme-toggle" 
            onClick={toggleTheme}
            aria-label="Toggle theme"
          >
            {theme === 'light' ? (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
              </svg>
            ) : (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="4" />
                <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41" />
              </svg>
            )}
          </button>
        </div>
      </header>

      {/* Main Grid */}
      <main className="grid">
        {/* Prediction */}
        <section className="card">
          <div className="card-head">
            <h2>Prediction</h2>
            <span className="card-tag">Inference</span>
          </div>
          <p className="card-sub">Run the model on a sample iris input.</p>
          <button className="btn btn-primary" onClick={makePrediction} disabled={loading}>
            {loading ? 'Running...' : 'Run Prediction'}
          </button>
          {prediction && (
            <div className="result-block">
              <div className="result-row">
                <span className="label">Class</span>
                <span className="value">{prediction.flower}</span>
              </div>
              <div className="result-row">
                <span className="label">Index</span>
                <span className="value mono">{prediction.prediction}</span>
              </div>
            </div>
          )}
        </section>

        {/* Drift */}
        <section className="card">
          <div className="card-head">
            <h2>Drift Monitor</h2>
            <span className="card-tag">Monitoring</span>
          </div>
          <p className="card-sub">PSI compared against threshold of 0.2.</p>
          
          {drift && (
            <div className="gauge-wrap">
              <svg viewBox="0 0 120 120" className="gauge">
                <circle cx="60" cy="60" r="50" className="gauge-track" />
                <circle
                  cx="60"
                  cy="60"
                  r="50"
                  className={`gauge-fill ${drift.drift_detected ? 'gauge-alert' : ''}`}
                  strokeDasharray={`${driftPercent * 3.14} 314`}
                />
              </svg>
              <div className="gauge-label">
                <span className="gauge-value mono">{drift.psi.toFixed(3)}</span>
                <span className="gauge-unit">PSI</span>
              </div>
            </div>
          )}

          {drift && (
            <div className={`status-line ${drift.drift_detected ? 'status-alert' : 'status-ok'}`}>
              {drift.drift_detected ? 'Drift detected' : 'No drift'}
            </div>
          )}

          <div className="btn-row">
            <button className="btn btn-secondary" onClick={addExtremeData}>
              Add Sample Data
            </button>
            <button className="btn btn-primary" onClick={checkDrift}>
              Refresh
            </button>
          </div>

          {drift?.drift_detected && (
            <button className="btn btn-alert" onClick={triggerRetrain}>
              Trigger Auto-Retrain
            </button>
          )}
        </section>

        {/* SHAP */}
        <section className="card">
          <div className="card-head">
            <h2>Explainability</h2>
            <span className="card-tag">SHAP</span>
          </div>
          <p className="card-sub">Feature contribution to the current prediction.</p>
          <button className="btn btn-primary" onClick={getExplanation}>
            Explain Prediction
          </button>

          {shap && (
            <div className="shap-list">
              {Object.entries(shap.shap_values || {}).map(([key, val]) => {
                const max = Math.max(
                  ...Object.values(shap.shap_values).map(v => Math.abs(v))
                )
                const width = max > 0 ? (Math.abs(val) / max) * 100 : 0
                const positive = val >= 0
                return (
                  <div className="shap-row" key={key}>
                    <span className="shap-label">{key.replace(/_/g, ' ')}</span>
                    <div className="shap-track">
                      <div
                        className={`shap-bar ${positive ? 'shap-pos' : 'shap-neg'}`}
                        style={{ width: `${width}%` }}
                      />
                    </div>
                    <span className="shap-value mono">
                      {val >= 0 ? '+' : ''}{val.toFixed(3)}
                    </span>
                  </div>
                )
              })}
            </div>
          )}
        </section>
      </main>

      {/* Footer */}
      <footer className="footer">
        <span>Built with FastAPI · MLflow · Evidently · SHAP</span>
      </footer>

      {/* Retrain Modal */}
      {retrainResult && (
        <div className="modal-overlay" onClick={() => setRetrainResult(null)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <h3>Auto-Retrain Complete</h3>
            <p className="modal-msg">{retrainResult.message}</p>
            {retrainResult.retrain_output && (
              <pre className="modal-pre">{retrainResult.retrain_output}</pre>
            )}
            <button className="btn btn-primary" onClick={() => setRetrainResult(null)}>
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

export default App