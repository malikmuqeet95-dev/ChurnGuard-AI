# ChurnGuard AI: 

## Production Customer Churn Survival & Retention Decision Intelligence System

* **Live Dashboard:** https://churnguard-ai-lovat.vercel.app

### 1. Executive Summary

ChurnGuard AI is an end-to-end machine learning and MLOps platform engineered for customer churn survival prediction and retention decision intelligence. The system integrates time-to-event survival analysis, directional explainability, counterfactual intervention simulation, ROI-based decision support, MLOps lifecycle governance, API security, Docker containerization, CI/CD automation, and operational observability.
Built as a production-grade, observable service, the platform is backed by automated CI/CD pipelines, container security hardening, in-process operational telemetry, and an automated test suite comprising passing unit and integration tests.

---

### 2. Core Functional Capabilities

Traditional churn systems answer only whether a customer will churn, whereas ChurnGuard AI estimates churn risk over time, identifies associated model signals, forecasts retention, and simulates intervention economics. The system features:  

- Time-to-event survival prediction 
- Churn-risk estimation
- Retention forecasting   
- Model explainability   
- Business rules integration   
- Intervention simulation 
- Intervention ROI estimation   
- Retention decision intelligence  
- Data-drift and model monitoring  
- Retraining management   
- Production REST API via FastAPI   
- Operational observability   
- Docker deployment and CI/CD 

---

### 3. High-Level Architecture Flow
```text 
        ┌────────────────────────────────────────────────────────┐
        │                   Frontend (Vercel)                    │
        │               HTML5 / CSS3 / Vanilla JS                │
        └───────────────────────────┬────────────────────────────┘
                                    │ HTTPS / JSON
                                    ▼
        ┌────────────────────────────────────────────────────────┐
        │                 FastAPI Backend (Render)               │
        │ ┌───────────────────┐        ┌───────────────────────┐ │
        │ │ Request Context   │        │ API-Key Auth          │ │
        │ │ Request ID Tracer │        │ Security Headers      │ │
        │ └─────────┬─────────┘        └───────────────────────┘ │
        │           ▼                                            │
        │ ┌────────────────────────────────────────────────────┐ │
        │ │ Pydantic Input Validation & Sanitization           │ │
        │ └─────────────────────────┬──────────────────────────┘ │
        │                           ▼                            │
        │ ┌────────────────────────────────────────────────────┐ │
        │ │ Prediction Service Orchestrator                    │ │
        │ └─────────────────────────┬──────────────────────────┘ │
        └───────────────────────────┼────────────────────────────┘
                                    ▼
        ┌────────────────────────────────────────────────────────┐
        │           Survival Engine (Cox PH / lifelines)         │
        │  * Hazard Multiplier  * Survival Curves  * Risk Tiers  │
        └───────────────────────────┬────────────────────────────┘
                                    ▼
        ┌────────────────────────────────────────────────────────┐
        │            Decision Intelligence Pipeline              │
        │                                                        │
        │ ┌───────────────────┐        ┌───────────────────────┐ │
        │ │ Model Signals /   │        │ Business Rule Engine  │ │
        │ │ Explainability    │        │ Hard Constraints      │ │
        │ └─────────┬─────────┘        └───────────┬───────────┘ │
        │           └───────────────┬──────────────┘             │
        │                           ▼                            │
        │                 Action Prioritization                  │
        │                           ▼                            │
        │                 Intervention Library                   │
        │                           ▼                            │
        │ ┌────────────────────────────────────────────────────┐ │
        │ │ Counterfactual Simulator & ROI Estimator           │ │
        │ └─────────────────────────┬──────────────────────────┘ │
        │                           ▼                            │
        │ ┌────────────────────────────────────────────────────┐ │
        │ │ Final Retention Decision & Action Tier             │ │
        │ └────────────────────────────────────────────────────┘ │
        └───────────────────────────┬────────────────────────────┘
                                    ▼
        ┌────────────────────────────────────────────────────────┐
        │               In-Process Metrics & Monitoring          │
        │   Latency (p95/p99) | Request Counts | Outcomes        │
        └────────────────────────────────────────────────────────┘
 ```

### 4. Technology Stack

- Machine Learning & Core Analytics: Python, pandas, NumPy, scikit-learn, lifelines (Cox Proportional Hazards).
- API & Backend Service: FastAPI, Pydantic, Uvicorn.
- Frontend Interface: Vanilla HTML5, CSS3, ES6 JavaScript.
- Testing & Verification: pytest, pytest-cov.
- Containerization & CI/CD: Docker, Docker Compose, GitHub Actions.

### 5. Limitations & Intended Use

- Decision Support Only: ChurnGuard AI provides decision-support analysis rather than automated, irreversible policy execution.
- Non-Causal Interpretability: Model-derived risk drivers and what-if simulations evaluate statistical associations within the trained distribution; they do not represent causal treatment effects.
- Financial Estimates: Projected ROI values are heuristic planning calculations dependent on user-supplied assumptions, not guaranteed business returns.
- Metrics Scope: Runtime operational metrics are collected in-memory per process and reset upon container restart.

### 6. Quickstart & Local Setup

```powershell
# 1. Clone & environment setup
git clone <repository-url>
cd customer_churn_survival_mlops
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt

# 2. Run automated test suite
python -m pytest tests/ -v

# 3. Launch local API server
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

### 7. Author
**Malik Muqeet**