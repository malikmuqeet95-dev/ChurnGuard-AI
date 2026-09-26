## ChurnGuard AI — System Architecture

### 1. Overview

ChurnGuard AI is an end-to-end customer churn survival and retention decision-intelligence system.

The architecture separates the system into several logical layers:

1. Data and ML layer
2. Decision-intelligence layer
3. API/application layer
4. Frontend layer
5. MLOps layer
6. Production/deployment layer
7. Observability layer

The separation allows each layer to be developed, tested, monitored, and maintained independently.

---

### 2. High-Level Architecture

```text
                         ┌─────────────────────────────┐
                         │       Customer/User         │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │       ChurnGuard UI         │
                         │        Frontend             │
                         └──────────────┬──────────────┘
                                        │ HTTP/JSON
                                        ▼
                 ┌──────────────────────────────────────────┐
                 │              FastAPI Backend             │
                 │                                          │
                 │  ┌────────────┐    ┌─────────────────┐   │
                 │  │ Validation │    │ Authentication  │   │
                 │  └─────┬──────┘    └─────────────────┘   │
                 │        │                                 │
                 │        ▼                                 │
                 │  ┌──────────────────────────────┐        │
                 │  │     Prediction Service       │        │
                 │  └──────────────┬───────────────┘        │
                 └─────────────────┼────────────────────────┘
                                   │
                                   ▼
                    ┌────────────────────────────┐
                    │     Survival Predictor     │
                    │       Cox PH Model         │
                    └──────────────┬─────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
             Risk Forecast   Survival Curve   Hazard Estimate
                    │
                    ▼
             ┌───────────────┐
             │ Explainability│
             └───────┬───────┘
                     │
                     ▼
             ┌───────────────┐
             │ Business Rules│
             └───────┬───────┘
                     │
                     ▼
             ┌───────────────┐
             │ Action Priority│
             └───────┬───────┘
                     │
                     ▼
          ┌────────────────────────┐
          │ Intervention Library   │
          └───────────┬────────────┘
                      │
              ┌───────┴────────┐
              ▼                ▼
       ┌─────────────┐  ┌──────────────┐
       │ Simulation  │  │ ROI Estimate │
       └──────┬──────┘  └──────┬───────┘
              │                │
              └───────┬────────┘
                      ▼
             ┌──────────────────┐
             │ Retention        │
             │ Decision Engine  │
             └────────┬─────────┘
                      │
                      ▼
             Decision-support output
```

### 3. MLOps lifecycle

The platform features an automated, auditable model lifecycle:
```text
        Raw Data ──► Data Validation ──► Dataset Versioning ──► Feature Pipeline
                                                                      │
        Production ◄── Model Promotion ◄── Quality Gate ◄── Training & Eval
            │                                (C-Index ≥ 0.90)
            ├──► Drift Monitoring
            ├──► Retraining Manager
            └──► Performance Tracking

```
