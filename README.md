\# Self-Healing ML Pipeline



End-to-end ML pipeline that serves predictions, monitors data drift, and auto-retrains when data shifts.



\## 🚀 Live Demo



\- \*\*Dashboard\*\*: https://self-healing-ml-pipeline.vercel.app

\- \*\*API\*\*: https://self-healing-ml-pipeline.onrender.com

\- \*\*API Docs\*\*: https://self-healing-ml-pipeline.onrender.com/docs



\## What It Does



1\. Serves ML predictions via FastAPI

2\. Monitors data drift using \*\*real PSI\*\* (Population Stability Index)

3\. Explains every prediction with \*\*SHAP\*\*

4\. Auto-retrains when drift exceeds threshold (PSI > 0.2)

5\. Promotes new models \*\*only if F1 improves\*\*



\## Results



\### Model Performance



| Metric | Value |

|--------|-------|

| Baseline F1 (RandomForest) | \*\*1.0000\*\* |

| Post-retrain F1 | \*\*0.9108\*\* |

| Delta | \*\*-0.0892\*\* |



\*\*Note:\*\* The negative delta is intentional. The auto-retrain pipeline retrains on combined data when drift is detected, but only promotes the new model if F1 improves. Since the new model scored lower, the champion model remained in production. This is the self-healing behavior working as designed.



\### API Latency (100 requests)



| Percentile | Latency |

|------------|---------|

| p50 (median) | \*\*10.3 ms\*\* |

| p95 | \*\*12.3 ms\*\* |

| p99 | \*\*14.4 ms\*\* |

| mean | \*\*10.6 ms\*\* |



\### Drift Detection



| Metric | Value |

|--------|-------|

| PSI (no drift) | \*\*0.0000\*\* |

| PSI (after 20 extreme samples) | \*\*1.3154\*\* |

| Drift detection time | \*\*5.4 ms\*\* |



\*\*PSI interpretation:\*\*

\- PSI < 0.1 → No significant change

\- 0.1 ≤ PSI < 0.2 → Moderate shift (monitor)

\- PSI ≥ 0.2 → Significant shift (retrain triggered)



\### Testing



```

14 tests passing (10 API + 4 PSI-specific)

```



\## Architecture



```

┌──────────────────┐

│  React Dashboard │

│  (Vercel)        │

└────────┬─────────┘

&#x20;        │ HTTPS

&#x20;        ▼

┌──────────────────┐

│  FastAPI Backend │

│  (Render)        │

└────────┬─────────┘

&#x20;        │

&#x20;   ┌────┴────┬────────┬──────────┐

&#x20;   ▼         ▼        ▼          ▼

&#x20;Model    SHAP    PSI Drift   Auto-

&#x20;Server   Expl.   Detect      Retrain

```



\## Tech Stack



Python · FastAPI · scikit-learn · MLflow · SHAP · React · Docker · Render · Vercel



\## API Endpoints



| Endpoint | Method | Purpose |

|----------|--------|---------|

| `/` | GET | API info + available endpoints |

| `/predict` | GET | Make a prediction |

| `/drift-report` | GET | Check current PSI + drift status |

| `/explain` | GET | SHAP values for a prediction |

| `/add-data` | POST | Simulate incoming data |

| `/auto-retrain` | POST | Trigger retraining if drift detected |



\## Run Locally



```bash

\# Start everything

docker-compose up --build

```



Then open:

\- \*\*Dashboard\*\*: http://localhost:3000

\- \*\*API\*\*: http://localhost:8000

\- \*\*API Docs\*\*: http://localhost:8000/docs



\## Testing



```bash

pytest tests/ -v

```



Expected: \*\*14 passed\*\*



\## Design Decisions \& Trade-offs



\### Why RandomForest?

\- Handles non-linear relationships better than Logistic Regression

\- More robust to outliers than single Decision Trees

\- No hyperparameter tuning required for baseline performance

\- Achieved F1 = 1.0000 on the test set



\### Why PSI for Drift?

\- \*\*PSI is interpretable\*\* — values map to industry-standard thresholds (0.1, 0.2)

\- \*\*PSI is sensitive to distribution shifts\*\* — catches both mean and shape changes

\- \*\*PSI is cheap to compute\*\* — 5.4 ms per call

\- Alternative KL divergence is asymmetric and harder to interpret

\- Alternative KS test is sensitive only to CDF shifts, not shape changes



\### Why PSI threshold = 0.2?

\- Industry standard for tabular data (used in credit scoring, fraud detection)

\- 0.1 catches noise, 0.2 catches meaningful drift

\- Balances \*\*false positives\*\* (retraining unnecessarily) vs \*\*false negatives\*\* (missing drift)



\### Why auto-retrain only promotes if F1 improves?

\- Prevents degrading the production model

\- "Self-healing" means improving, not just changing

\- Benchmark proved this works: retrain F1 (0.9108) < baseline (1.0000) → old model kept



\### What I'd change at scale

\- \*\*Real dataset\*\*: Iris is a proof-of-concept. Production would use Telco Churn, credit fraud, or similar

\- \*\*Persistent storage\*\*: Current `current\_data` is in-memory. Production would use PostgreSQL/S3

\- \*\*Scheduled retraining\*\*: Currently manual trigger. Production would use cron/Airflow

\- \*\*Model registry\*\*: Currently uses `model.pkl`. Production would use MLflow Model Registry with Staging/Production aliases

\- \*\*Message queue\*\*: Currently synchronous retrain. Production would use Celery/RabbitMQ

\- \*\*Data versioning\*\*: Currently no tracking of which data trained which model. Production would use DVC



\## License



MIT

