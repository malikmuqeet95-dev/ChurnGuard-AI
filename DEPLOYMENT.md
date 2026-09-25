# ChurnGuard AI — Production Deployment Guide

## 1. Overview

ChurnGuard AI is a customer churn survival and retention decision-intelligence
system built around a Cox Proportional Hazards survival model.

The production architecture consists of:

- FastAPI backend
- Cox survival prediction engine
- Explainability layer
- Business rules
- Intervention simulation
- Retention decision engine
- MLOps lifecycle
- Docker production container
- CI/CD workflows
- Operational metrics
- Health and readiness monitoring
- API-key protection

---

## 2. Production Architecture

```text
Client / Frontend
        |
        v
   FastAPI API
        |
        +------------------+
        |                  |
        v                  v
 Prediction            What-if
 Service               Analysis
        |                  |
        +--------+---------+
                 |
                 v
        Survival Predictor
                 |
                 v
             CoxPH Model
                 |
                 v
        Retention Decision
                 |
                 v
       Operational Metrics