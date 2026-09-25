/* ============================================================
   CHURNGUARD AI
   FRONTEND APPLICATION
   PHASE 5 - FOUNDATION + DASHBOARD SHELL
   ============================================================ */


/* ============================================================
   CONFIGURATION
   ============================================================ */

/*
    You can override the API URL from the browser console with:

    window.CHURNGUARD_API_BASE_URL = "http://127.0.0.1:8000";

    The default below is appropriate for local development.
*/

const API_BASE_URL =
    window.CHURNGUARD_API_BASE_URL ||
    "http://127.0.0.1:8000";


/* ============================================================
   DOM HELPERS
   ============================================================ */

const $ = (selector) => document.querySelector(selector);

const $$ = (selector) =>
    Array.from(document.querySelectorAll(selector));


/* ============================================================
   DOM REFERENCES
   ============================================================ */

const predictionForm = $("#prediction-form");

const predictButton = $("#predict-button");
const predictButtonText = $("#predict-button-text");
const predictSpinner = $("#predict-spinner");

const resetButton = $("#reset-button");

const globalAlert = $("#global-alert");
const globalAlertMessage = $("#global-alert-message");
const closeAlertButton = $("#close-alert");

const apiStatusDot = $("#api-status-dot");
const apiStatusText = $("#api-status-text");

const requestIdDisplay = $("#request-id-display");


/* ============================================================
   APPLICATION STATE
   ============================================================ */

const state = {
    lastPrediction: null,
    apiOnline: false,
    isPredicting: false
};


/* ============================================================
   DEFAULT CUSTOMER
   ============================================================ */

const DEFAULT_CUSTOMER = {
    tenure: 12,
    forecast_horizon: 12,

    gender: "Male",
    SeniorCitizen: 0,
    Partner: "No",
    Dependents: "No",

    MonthlyCharges: 79.85,
    TotalCharges: 958.20,

    Contract: "Month-to-month",
    PaperlessBilling: "Yes",
    PaymentMethod: "Electronic check",

    PhoneService: "Yes",
    MultipleLines: "No",

    InternetService: "Fiber optic",

    OnlineSecurity: "No",
    OnlineBackup: "No",
    DeviceProtection: "No",
    TechSupport: "No",

    StreamingTV: "No",
    StreamingMovies: "No"
};


/* ============================================================
   INITIALIZATION
   ============================================================ */

document.addEventListener("DOMContentLoaded", () => {

    initializeNavigation();

    initializeForm();

    initializeAlert();

    applyDefaultCustomer();

    checkApiHealth();
    initializeSimulation();

});


/* ============================================================
   NAVIGATION
   ============================================================ */

function initializeNavigation() {

    const navLinks =
        document.querySelectorAll(
            ".sidebar-nav a[href^='#']"
        );

    const sections =
        document.querySelectorAll(
            "main section[id]"
        );

    if (!navLinks.length || !sections.length) {
        return;
    }

    navLinks.forEach(link => {

        link.addEventListener(
            "click",
            event => {

                const targetId =
                    link.getAttribute("href");

                if (!targetId) {
                    return;
                }

                const target =
                    document.querySelector(targetId);

                if (!target) {
                    return;
                }

                event.preventDefault();

                target.scrollIntoView({
                    behavior: "smooth",
                    block: "start"
                });

                navLinks.forEach(item => {
                    item.classList.remove("active");
                });

                link.classList.add("active");
            }
        );
    });


    const observer =
        new IntersectionObserver(
            entries => {

                entries.forEach(entry => {

                    if (!entry.isIntersecting) {
                        return;
                    }

                    const id =
                        entry.target.id;

                    navLinks.forEach(link => {

                        const target =
                            link.getAttribute("href");

                        link.classList.toggle(
                            "active",
                            target === `#${id}`
                        );
                    });
                });

            },
            {
                rootMargin:
                    "-20% 0px -65% 0px",

                threshold: 0
            }
        );


    sections.forEach(section => {
        observer.observe(section);
    });
}


/* ============================================================
   FORM INITIALIZATION
   ============================================================ */

function initializeForm() {
    if (!predictionForm) {
        return;
    }

    predictionForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        event.stopPropagation();
        await runPrediction();
    });

    resetButton.addEventListener("click", () => {
        applyDefaultCustomer();
        clearPredictionResults();
        hideAlert();
    });
}


/* ============================================================
   ALERT INITIALIZATION
   ============================================================ */

function initializeAlert() {

    if (!closeAlertButton) {
        return;
    }


    closeAlertButton.addEventListener(
        "click",
        hideAlert
    );

}


/* ============================================================
   APPLY DEFAULT CUSTOMER
   ============================================================ */

function applyDefaultCustomer() {

    Object.entries(DEFAULT_CUSTOMER)
        .forEach(([name, value]) => {

            const element =
                document.querySelector(
                    `[name="${name}"]`
                );

            if (!element) {
                return;
            }

            element.value = value;

        });

}


/* ============================================================
   API HEALTH
   ============================================================ */

async function checkApiHealth() {
    setApiStatus("checking", "Checking API...");

    try {
        const response = await fetch(`${API_BASE_URL}/health`, {
            method: "GET",
            headers: { "Accept": "application/json" }
        });

        if (!response.ok) {
            throw new Error(`Health check failed (${response.status})`);
        }

        const data = await response.json();
        if (data.status === "ok") {
            state.apiOnline = true;
            setApiStatus("online", "API Online");
            return true;
        }

        throw new Error("Unexpected health check status.");
    } catch (error) {
        state.apiOnline = false;
        setApiStatus("offline", "API Offline");
        console.warn("API health check failed:", error);
        return false;
    }
}

/* ============================================================
   API STATUS UI
   ============================================================ */

function setApiStatus(status, text) {
    if (!apiStatusDot) return;

    apiStatusDot.className = "status-dot";

    if (status === "online") {
        apiStatusDot.classList.add("status-online", "api-online");
    } else if (status === "offline") {
        apiStatusDot.classList.add("status-offline", "api-offline");
    } else {
        apiStatusDot.classList.add("status-checking", "api-checking");
    }

    if (apiStatusText) {
        apiStatusText.textContent = text || (
            status === "online"
                ? "API Online"
                : status === "checking"
                    ? "Checking API..."
                    : "API Offline"
        );
    }
}

/* ============================================================
   FORM -> PAYLOAD
   ============================================================ */

function buildPredictionPayload() {

    const formData =
        new FormData(predictionForm);


    const payload = {};


    formData.forEach((value, key) => {

        if (
            [
                "tenure",
                "forecast_horizon",
                "SeniorCitizen",
                "MonthlyCharges",
                "TotalCharges"
            ].includes(key)
        ) {

            payload[key] = Number(value);

        } else {

            payload[key] = value;

        }

    });


    return payload;

}

/* ============================================================
   USER-FRIENDLY ERROR HELPER
   ============================================================ */

function getUserFriendlyError(error) {
    if (!error) {
        return "An unexpected error occurred.";
    }

    const message = String(error.message || error);

    if (message.includes("Failed to fetch") || message.includes("NetworkError")) {
        return "Unable to reach the ChurnGuard API. Make sure the FastAPI backend is running.";
    }

    if (message.includes("422")) {
        return "The customer data failed validation. Please review the entered values.";
    }

    if (message.includes("500")) {
        return "The prediction service encountered an internal error. Please try again.";
    }

    return message;
}

/* ============================================================
   RUN PREDICTION
   ============================================================ */

async function runPrediction() {

    if (state.isPredicting) {
        return;
    }


    hideAlert();

    setPredictionLoading(true);


    const payload =
        buildPredictionPayload();


    try {

        const response =
            await fetch(
                `${API_BASE_URL}/api/predict`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json",

                        "Accept":
                            "application/json"
                    },

                    body:
                        JSON.stringify(payload)
                }
            );


        const requestId =
            response.headers.get(
                "x-request-id"
            );


        if (requestId) {

            requestIdDisplay.textContent =
                `Request: ${requestId}`;

        }


        const data =
            await parseJsonResponse(
                response
            );


        if (!response.ok) {

            throw new Error(
                extractApiError(data)
            );

        }


        state.lastPrediction = data;

        renderPrediction(data);


        state.apiOnline = true;

        setApiStatus(
            "online",
            "API Online"
        );


        /*
            Move the user to the risk overview
            after a successful analysis.
        */

        setTimeout(() => {

            document
                .querySelector("#risk-overview")
                ?.scrollIntoView({
                    behavior: "smooth",
                    block: "start"
                });

        }, 150);

    } catch (error) {

        console.error(
            "Prediction request failed:",
            error
        );


        showAlert(getUserFriendlyError(error));

        if (
            error.message?.includes(
                "Failed to fetch"
            )
        ) {

            state.apiOnline = false;

            setApiStatus(
                "offline",
                "API Offline"
            );

        }

    } finally {

        setPredictionLoading(false);

    }

}


/* ============================================================
   PARSE API RESPONSE
   ============================================================ */

async function parseJsonResponse(response) {

    const text =
        await response.text();


    if (!text) {
        return {};
    }


    try {

        return JSON.parse(text);

    } catch {

        return {
            detail:
                text
        };

    }

}


/* ============================================================
   API ERROR EXTRACTION
   ============================================================ */

function extractApiError(data) {

    if (!data) {
        return "Unknown API error.";
    }


    if (typeof data.detail === "string") {
        return data.detail;
    }


    if (Array.isArray(data.detail)) {

        return data.detail
            .map((item) => {

                if (
                    typeof item === "string"
                ) {
                    return item;
                }

                return (
                    item.msg ||
                    "Validation error"
                );

            })
            .join("; ");

    }


    if (typeof data.message === "string") {
        return data.message;
    }


    return "The API returned an unexpected error.";

}

/* ============================================================
   LOADING STATE
   ============================================================ */

function setPredictionLoading(isLoading) {
    state.isPredicting = isLoading;

    if (predictButton) {
        predictButton.disabled = isLoading;
        predictButton.classList.toggle("is-loading", isLoading);
    }

    if (resetButton) {
        resetButton.disabled = isLoading;
    }

    if (isLoading) {
        if (predictButtonText) predictButtonText.textContent = "Analyzing...";
        if (predictSpinner) predictSpinner.classList.remove("hidden");
    } else {
        if (predictButtonText) predictButtonText.textContent = "Analyze Customer";
        if (predictSpinner) predictSpinner.classList.add("hidden");
    }
}


/* ============================================================
   STEP 5: WHAT-IF INTERVENTION SIMULATOR
   ============================================================ */

const SIMULATION_FIELDS = {
    "sim-contract": "Contract",
    "sim-payment": "PaymentMethod",
    "sim-tech-support": "TechSupport",
    "sim-security": "OnlineSecurity"
};

function getSimulationChanges() {
    const changes = {};

    Object.entries(SIMULATION_FIELDS).forEach(([elementId, fieldName]) => {
        const element = document.getElementById(elementId);
        if (element && element.value && element.value !== "") {
            changes[fieldName] = element.value;
        }
    });

    return changes;
}

function resetWhatIfSimulation() {
    Object.keys(SIMULATION_FIELDS).forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.value = "";
        }
    });

    const result = document.getElementById("simulation-result");
    if (result) {
        result.innerHTML = `
            <div class="simulation-empty">
                <div class="empty-icon">↗</div>
                <h3>No simulation yet</h3>
                <p>
                    Select one or more changes
                    and run a what-if simulation.
                </p>
            </div>
        `;
    }
}

async function runWhatIfSimulation() {
    const resultContainer = document.getElementById("simulation-result");
    if (!resultContainer) return;

    // Use your existing payload builder function
    const currentCustomer = buildPredictionPayload();

    if (!currentCustomer || !state.lastPrediction) {
        resultContainer.innerHTML = `
            <div class="simulation-error" style="color: #dc2626; padding: 12px; background: #fef2f2; border-radius: 6px; font-size: 12px;">
                Run a customer analysis first before simulating scenarios.
            </div>
        `;
        return;
    }

    const changes = getSimulationChanges();

    if (!Object.keys(changes).length) {
        resultContainer.innerHTML = `
            <div class="simulation-error" style="color: #dc2626; padding: 12px; background: #fef2f2; border-radius: 6px; font-size: 12px;">
                Select at least one change to simulate.
            </div>
        `;
        return;
    }

    resultContainer.innerHTML = `
        <div class="simulation-loading" style="color: #1f6feb; padding: 16px; font-size: 13px; font-weight: 600;">
            Running model-based simulation...
        </div>
    `;

    try {
        // Pointing to your FastAPI route /api/predict
        const response = await fetch(`${API_BASE_URL}/api/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            body: JSON.stringify({
                ...currentCustomer,
                ...changes
            })
        });

        if (!response.ok) {
            throw new Error(`Simulation failed (${response.status})`);
        }

        const simulated = await response.json();
        renderSimulationResult(simulated, currentCustomer, changes);

    } catch (error) {
        console.error("Simulation error:", error);
        resultContainer.innerHTML = `
            <div class="simulation-error" style="color: #dc2626; padding: 12px; background: #fef2f2; border-radius: 6px; font-size: 12px;">
                ${escapeHtml(error.message)}
            </div>
        `;
    }
}

function initializeSimulation() {
    const simulationButton = document.getElementById("run-simulation");
    if (simulationButton) {
        simulationButton.addEventListener("click", runWhatIfSimulation);
    }

    const resetSimulationButton = document.getElementById("reset-simulation");
    if (resetSimulationButton) {
        resetSimulationButton.addEventListener("click", resetWhatIfSimulation);
    }
}

/* ============================================================
   RENDER PREDICTION
   ============================================================ */

function renderPrediction(data) {

    renderRiskOverview(data);
    renderForecast(data);
    renderForecastSummary(data);
    renderInterventions(data);
    renderEconomics(data);
    renderExplainability(data);
    renderDecision(data);

    resetWhatIfSimulation();

}


/* ============================================================
   RISK OVERVIEW
   ============================================================ */

function renderRiskOverview(data) {
    setText("#projected-churn", formatPercent(data.projected_churn));
    setText("#projected-retention", formatPercent(data.projected_retention));
    setText("#hazard-multiplier", formatMultiplier(data.hazard_ratio));

    const riskTier = normalizeRiskTier(data.risk_tier);
    const riskElement = $("#risk-tier");

    if (riskElement) {
        riskElement.textContent = riskTier || "UNKNOWN";
        riskElement.className = "risk-badge";

        if (riskTier === "CRITICAL") {
            riskElement.classList.add("risk-critical");
        } else if (riskTier === "HIGH") {
            riskElement.classList.add("risk-high");
        } else if (riskTier === "MODERATE") {
            riskElement.classList.add("risk-moderate");
        } else if (riskTier === "LOW" || riskTier === "HEALTHY") {
            riskElement.classList.add("risk-healthy");
        } else {
            riskElement.classList.add("risk-neutral");
        }
    }

    setText("#risk-action", data.recommended_action || "Review retention recommendation.");
}

/* ============================================================
   FORECAST SUMMARY (STEP 4)
   ============================================================ */

function renderForecastSummary(data) {
    const currentTenure = document.getElementById("forecast-current-tenure");
    const horizon = document.getElementById("forecast-horizon");
    const targetMonth = document.getElementById("forecast-target-month");
    const targetRetention = document.getElementById("forecast-target-retention");

    if (currentTenure) currentTenure.textContent = `${data.current_tenure} months`;
    if (horizon) horizon.textContent = `${data.forecast_horizon} months`;
    if (targetMonth) targetMonth.textContent = `Month ${data.target_month}`;
    if (targetRetention) targetRetention.textContent = formatPercent(data.projected_retention);
}

/* ============================================================
   FORECAST TIMELINE (STEP 4)
   ============================================================ */

function renderForecast(data) {
    const chart = document.getElementById("forecast-chart");
    if (!chart) return;

    if (!window.Plotly) {
        chart.innerHTML = `<p class="empty-state">Plotly could not be loaded.</p>`;
        return;
    }

    const curve = data.forecast_curve || data.survival_curve;
    if (!curve || !curve.timeline || !curve.survival_prob) {
        chart.className = "chart-placeholder";
        chart.innerHTML = `
            <div class="placeholder-icon">↗</div>
            <h4>Forecast visualization</h4>
            <p>Run a customer analysis to populate the survival forecast.</p>
        `;
        return;
    }

    // Remove placeholder flex styling so Plotly gets full container dimensions
    chart.className = "";
    chart.style.width = "100%";
    chart.style.minHeight = "360px";

    const currentTenure = Number(data.current_tenure ?? data.tenure ?? 12);
    const horizon = Number(data.forecast_horizon ?? 12);
    const targetMonth = Number(data.target_month ?? (currentTenure + horizon));

    // Slice curve to focus on the active forecast window (current_tenure to target_month)
    const allMonths = curve.timeline.map(Number);
    const allSurvival = curve.survival_prob.map(p => normalizeProbability(p) * 100);
    const allChurn = curve.churn_prob
        ? curve.churn_prob.map(p => normalizeProbability(p) * 100)
        : allSurvival.map(s => Math.max(0, 100 - s));

    // Filter points for the relevant horizon window
    let displayIndices = allMonths
        .map((m, idx) => ({ m, idx }))
        .filter(item => item.m >= currentTenure && item.m <= targetMonth)
        .map(item => item.idx);

    // Fallback: If current tenure slice yields fewer than 2 points, show initial window
    if (displayIndices.length < 2) {
        displayIndices = allMonths
            .map((m, idx) => ({ m, idx }))
            .filter(item => item.m <= targetMonth)
            .map(item => item.idx);
    }

    const months = displayIndices.map(i => allMonths[i]);
    const retention = displayIndices.map(i => allSurvival[i]);
    const churn = displayIndices.map(i => allChurn[i]);

    const targetIndex = months.findIndex(m => m === targetMonth);

    const traces = [
        {
            x: months,
            y: retention,
            mode: "lines+markers",
            name: "Retention Probability",
            line: { color: "#1f6feb", width: 3 },
            marker: { size: 6, color: "#1f6feb" },
            hovertemplate: "Month %{x}<br>Retention: %{y:.1f}%<extra></extra>"
        },
        {
            x: months,
            y: churn,
            mode: "lines+markers",
            name: "Churn Probability",
            line: { color: "#e05252", width: 2, dash: "dot" },
            marker: { size: 5, color: "#e05252" },
            hovertemplate: "Month %{x}<br>Churn: %{y:.1f}%<extra></extra>"
        }
    ];

    if (targetIndex >= 0) {
        traces.push({
            x: [months[targetIndex]],
            y: [retention[targetIndex]],
            mode: "markers",
            name: "Target Horizon",
            marker: { size: 12, color: "#36c991", symbol: "diamond" },
            hovertemplate: "Target Month %{x}<br>Retention: %{y:.1f}%<extra></extra>"
        });
    }

    const layout = {
        title: { text: "Modeled Retention Forecast", font: { size: 15, color: "#182230" } },
        xaxis: { title: "Tenure Month", dtick: 1, gridcolor: "#edf1f6" },
        yaxis: { title: "Probability (%)", range: [0, 105], gridcolor: "#edf1f6" },
        hovermode: "x unified",
        margin: { l: 60, r: 30, t: 45, b: 45 },
        legend: { orientation: "h", y: -0.2 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        autosize: true
    };

    chart.innerHTML = "";
    Plotly.newPlot(chart, traces, layout, { responsive: true, displayModeBar: false });
}

/* ============================================================
   SURVIVAL CURVE EXTRACTION
   ============================================================ */

function extractSurvivalPoints(curve) {

    const x = [];
    const y = [];


    curve.forEach((point, index) => {

        if (
            Array.isArray(point) &&
            point.length >= 2
        ) {

            x.push(point[0]);
            y.push(normalizeProbability(point[1]));

            return;
        }


        if (
            typeof point !== "object" ||
            point === null
        ) {
            return;
        }


        const xValue =
            firstDefined(
                point.month,
                point.tenure_months,
                point.target_month,
                point.time,
                point.x,
                index + 1
            );


        const yValue =
            firstDefined(
                point.survival,
                point.retention,
                point.survival_probability,
                point.retention_probability,
                point.y
            );


        if (
            yValue === undefined ||
            yValue === null
        ) {
            return;
        }


        x.push(Number(xValue));

        y.push(
            normalizeProbability(
                yValue
            )
        );

    });


    return {
        x,
        y
    };

}


/* ============================================================
   EXPLAINABILITY - BASIC SHELL
   ============================================================ */

function renderExplainability(data) {
    const riskContainer = $("#risk-drivers");
    const protectiveContainer = $("#protective-drivers");

    if (!riskContainer || !protectiveContainer) return;

    const explanation = data.explanation || {};
    const riskDrivers = explanation.risk_drivers || [];
    const protectiveDrivers = explanation.protective_drivers || [];

    const driverHtml = (driver, typeClass) => {
        const impact = safeNumber(driver.contribution);
        return `
            <div class="driver-card ${typeClass}" style="padding: 12px; margin-bottom: 10px; border-radius: 8px; border: 1px solid #dfe6ef; background: #ffffff;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <strong style="color: #1f2937; font-size: 13px;">${escapeHtml(driver.feature)}</strong>
                    <span style="font-weight: 700; font-size: 12px; color: ${impact >= 0 ? '#dc2626' : '#16a34a'};">
                        ${impact >= 0 ? '+' : ''}${impact.toFixed(3)}
                    </span>
                </div>
                <p style="margin: 0; font-size: 11px; color: #64748b;">${escapeHtml(driver.description)}</p>
                <small style="display: block; margin-top: 6px; font-size: 9px; color: #94a3b8; text-transform: uppercase;">
                    Model signal — not a causal effect.
                </small>
            </div>
        `;
    };

    riskContainer.innerHTML = Array.isArray(riskDrivers) && riskDrivers.length > 0
        ? riskDrivers.slice(0, 6).map(d => driverHtml(d, "risk-driver")).join("")
        : `<div class="empty-state"><span>—</span><p>No major risk drivers identified.</p></div>`;

    protectiveContainer.innerHTML = Array.isArray(protectiveDrivers) && protectiveDrivers.length > 0
        ? protectiveDrivers.slice(0, 6).map(d => driverHtml(d, "protective-driver")).join("")
        : `<div class="empty-state"><span>—</span><p>No protective signals identified.</p></div>`;
}


/* ============================================================
   DRIVER LIST
   ============================================================ */

function renderDriverList(container, drivers) {

    if (!container) return;

    if (!Array.isArray(drivers) || drivers.length === 0) {
        renderEmptyDriverState(container, "No driver information available.");
        return;
    }

    container.innerHTML = "";

    drivers.slice(0, 6).forEach((driver) => {
        const row = document.createElement("div");
        row.className = "driver-row";

        const name = document.createElement("strong");
        name.textContent =
            driver.feature ||
            driver.model_feature ||
            driver.name ||
            "Model signal";

        const description = document.createElement("span");
        description.textContent =
            driver.description ||
            (driver.hazard_multiplier ? `${Number(driver.hazard_multiplier).toFixed(2)}x hazard` : driver.impact) ||
            "Associated model signal";

        if (driver.direction === "increases_risk") {
            description.style.color = "#ef4444";
        } else if (driver.direction === "decreases_risk") {
            description.style.color = "#10b981";
        }

        row.appendChild(name);
        row.appendChild(description);
        container.appendChild(row);
    });
}


/* ============================================================
   EMPTY DRIVER STATE
   ============================================================ */

function renderEmptyDriverState(
    container,
    message
) {

    container.innerHTML = "";

    const wrapper =
        document.createElement("div");

    wrapper.className =
        "empty-state";


    const icon =
        document.createElement("span");

    icon.textContent = "—";


    const text =
        document.createElement("p");

    text.textContent =
        message;


    wrapper.appendChild(icon);

    wrapper.appendChild(text);

    container.appendChild(wrapper);

}


/* ============================================================
   INTERVENTION INTELLIGENCE RENDERER
   ============================================================ */

function renderInterventions(data) {
    const container = document.getElementById("interventions-content");
    if (!container) return;

    const decision = data.retention_decision || data.decision;
    const interventions = decision?.evaluated_interventions || [];

    if (!interventions.length) {
        container.innerHTML = `
            <div class="empty-state">
                <h3>No targeted interventions</h3>
                <p>The current customer profile does not require a targeted retention action.</p>
            </div>
        `;
        return;
    }

    const cards = interventions
        .sort((a, b) => safeNumber(a.decision_rank, 999) - safeNumber(b.decision_rank, 999))
        .map(intervention => {
            const roi = intervention.roi || {};
            const priority = intervention.priority || "STANDARD";
            const roiValue = safeNumber(roi.estimated_roi_pct, 0);
            const netValue = safeNumber(roi.estimated_net_value, 0);
            const improvement = safeNumber(roi.modeled_churn_improvement_pp, 0);

            return `
                <article class="intervention-card">
                    <div>
                        <div class="intervention-card-top">
                            <span class="intervention-rank">
                                #${safeNumber(intervention.decision_rank || intervention.rank, 0)}
                            </span>
                            <span class="priority-badge priority-${escapeHtml(priority.toLowerCase())}">
                                ${escapeHtml(priority)}
                            </span>
                        </div>

                        <h3>
                            ${escapeHtml(intervention.name || intervention.intervention_id)}
                        </h3>

                        <p class="intervention-category">
                            ${escapeHtml(intervention.category || "Retention")}
                        </p>

                        <div class="intervention-reason">
                            <strong>Why this action?</strong>
                            <p>
                                ${escapeHtml(intervention.reason || "Business rule or model signal supports this action.")}
                            </p>
                        </div>

                        <div class="intervention-meta">
                            <div>
                                <span>Channel</span>
                                <strong>${escapeHtml(intervention.channel || "N/A")}</strong>
                            </div>
                            <div>
                                <span>Objective</span>
                                <strong>${escapeHtml(intervention.objective || "N/A")}</strong>
                            </div>
                            <div>
                                <span>Eligibility</span>
                                <strong>${escapeHtml(intervention.eligibility || "N/A")}</strong>
                            </div>
                        </div>

                        <div class="intervention-economics">
                            <div>
                                <span>Churn Improvement</span>
                                <strong>${improvement.toFixed(2)} pp</strong>
                            </div>
                            <div>
                                <span>Net Value</span>
                                <strong>${formatCurrency(netValue)}</strong>
                            </div>
                            <div>
                                <span>ROI</span>
                                <strong>${roiValue.toFixed(1)}%</strong>
                            </div>
                        </div>
                    </div>

                    <div class="intervention-action">
                        <strong>Recommended Action</strong>
                        <p>
                            ${escapeHtml(intervention.recommended_action || "Review intervention with retention team.")}
                        </p>
                    </div>
                </article>
            `;
        })
        .join("");

    container.innerHTML = `
        <div class="intervention-grid">
            ${cards}
        </div>
    `;
}


/* ============================================================
   ECONOMICS - BASIC SHELL
   ============================================================ */
function renderEconomics(data) {

    const decision =
        data.retention_decision ||
        data.decision;

    const selected =
        decision?.selected_intervention;

    const roi =
        selected?.roi;

    const customerValue =
        document.getElementById(
            "customer-value"
        );

    const valuePreserved =
        document.getElementById(
            "value-preserved"
        );

    const netValue =
        document.getElementById(
            "net-value"
        );

    const roiPercent =
        document.getElementById(
            "roi-percent"
        );

    const recommendation =
        document.getElementById(
            "roi-recommendation"
        );

    const intervention =
        document.getElementById(
            "roi-intervention"
        );


    if (!roi) {

        if (customerValue) {
            customerValue.textContent = "—";
        }

        if (valuePreserved) {
            valuePreserved.textContent = "—";
        }

        if (netValue) {
            netValue.textContent = "—";
        }

        if (roiPercent) {
            roiPercent.textContent = "—";
        }

        if (recommendation) {
            recommendation.textContent =
                "No intervention";
        }

        if (intervention) {

            intervention.innerHTML = `
                <div class="empty-state">
                    No selected intervention is
                    currently available.
                </div>
            `;
        }

        return;
    }


    if (customerValue) {

        customerValue.textContent =
            formatCurrency(
                roi.customer_value_assumption
            );
    }


    if (valuePreserved) {

        valuePreserved.textContent =
            formatCurrency(
                roi.estimated_value_preserved
            );
    }


    if (netValue) {

        netValue.textContent =
            formatCurrency(
                roi.estimated_net_value
            );
    }


    if (roiPercent) {

        roiPercent.textContent =
            `${safeNumber(
                roi.estimated_roi_pct
            ).toFixed(1)}%`;
    }


    if (recommendation) {

        recommendation.textContent =
            roi.economic_recommendation ||
            "Review";
    }


    if (intervention) {

        intervention.innerHTML = `

            <div class="roi-intervention-card">

                <h4>
                    ${escapeHtml(
                        selected.name ||
                        selected.intervention_id
                    )}
                </h4>

                <p>
                    ${escapeHtml(
                        selected.recommended_action ||
                        ""
                    )}
                </p>


                <div class="roi-details">

                    <div>
                        <span>
                            Estimated Cost
                        </span>

                        <strong>
                            ${formatCurrency(
                                roi.estimated_cost
                            )}
                        </strong>
                    </div>


                    <div>
                        <span>
                            Modeled Churn Improvement
                        </span>

                        <strong>
                            ${safeNumber(
                                roi.modeled_churn_improvement_pp
                            ).toFixed(2)} pp
                        </strong>
                    </div>


                    <div>
                        <span>
                            Net Value
                        </span>

                        <strong>
                            ${formatCurrency(
                                roi.estimated_net_value
                            )}
                        </strong>
                    </div>


                    <div>
                        <span>
                            ROI
                        </span>

                        <strong>
                            ${safeNumber(
                                roi.estimated_roi_pct
                            ).toFixed(1)}%
                        </strong>
                    </div>

                </div>

            </div>

        `;
    }
}

/* ============================================================
   EXECUTIVE RETENTION MEMO RENDERER (SECTION 08)
   ============================================================ */

function renderDecision(data) {
    const container = document.getElementById("decision-content");
    if (!container) return;

    const decision = data.retention_decision || data.decision;

    if (!decision) {
        container.innerHTML = `
            <div class="empty-state">
                <h3>No decision generated</h3>
                <p>Run a customer prediction to generate retention decision intelligence.</p>
            </div>
        `;
        return;
    }

    const riskTier = String(decision.risk_tier || data.risk_tier || "MODERATE").toUpperCase();
    const priority = String(decision.operational_priority || "STANDARD").toUpperCase();
    const projectedChurn = safeNumber(decision.projected_churn ?? data.projected_churn_pct);
    const selected = decision.selected_intervention;
    const roi = selected?.roi || {};

    // Map raw policy names to polished human titles
    const recommendationMap = {
        "RECOMMEND_INTERVENTION": "Targeted Retention Intervention",
        "STANDARD_MONITORING": "Standard Account Monitoring",
        "HIGH_PRIORITY_OUTREACH": "Urgent Executive Outreach",
        "NO_ACTION": "Preserve Standard Workflow"
    };
    const rawRec = decision.recommendation || "RECOMMEND_INTERVENTION";
    const cleanRecommendation = recommendationMap[rawRec] || rawRec.replace(/_/g, " ");

    // Color definitions based on risk severity
    const isUrgent = priority.includes("URGENT") || riskTier.includes("CRITICAL") || riskTier.includes("HIGH");
    const accentColor = isUrgent ? "#dc2626" : "#2563eb";
    const accentBg = isUrgent ? "#fef2f2" : "#eff6ff";
    const accentBorder = isUrgent ? "#fecaca" : "#bfdbfe";

    container.innerHTML = `
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);">
            
            <!-- Top Command Header -->
            <div style="background: #0f172a; color: #ffffff; padding: 18px 24px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                <div>
                    <span style="font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; color: #94a3b8; font-weight: 700; display: block;">
                        Operational Protocol Verdict
                    </span>
                    <h3 style="margin: 2px 0 0 0; font-size: 20px; font-weight: 700; color: #f8fafc;">
                        ${escapeHtml(cleanRecommendation)}
                    </h3>
                </div>

                <!-- Status Pills -->
                <div style="display: flex; gap: 8px; align-items: center;">
                    <span style="background: ${accentBg}; color: ${accentColor}; border: 1px solid ${accentBorder}; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 800; letter-spacing: 0.04em;">
                        ${escapeHtml(priority)} PRIORITY
                    </span>
                    <span style="background: #1e293b; color: #e2e8f0; border: 1px solid #334155; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600;">
                        ${formatPercent(projectedChurn)} Horizon Churn
                    </span>
                </div>
            </div>

            <!-- Body Details -->
            <div style="padding: 24px; display: flex; flex-direction: column; gap: 20px;">
                
                <!-- Actionable Playbook Banner -->
                ${selected ? `
                    <div style="background: #f8fafc; border-left: 4px solid ${accentColor}; border-top: 1px solid #e2e8f0; border-right: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0; border-radius: 0 8px 8px 0; padding: 18px;">
                        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px;">
                            <span style="font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;">
                                Recommended Playbook
                            </span>
                            <span style="font-size: 12px; color: #475569;">
                                Channel: <strong style="color: #0f172a;">${escapeHtml(selected.channel || "Customer Success")}</strong>
                            </span>
                        </div>

                        <h4 style="margin: 0 0 8px 0; font-size: 17px; color: #0f172a; font-weight: 700;">
                            ${escapeHtml(selected.name || selected.intervention_id)}
                        </h4>
                        
                        <p style="margin: 0; font-size: 13.5px; color: #334155; line-height: 1.5;">
                            ${escapeHtml(selected.recommended_action || "Deploy targeted retention incentive outreach.")}
                        </p>
                    </div>
                ` : `
                    <div style="background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 8px; padding: 16px; color: #64748b; font-size: 13px;">
                        No targeted counterfactual intervention selected. Continue routine retention monitoring.
                    </div>
                `}

                <!-- Rationale Paragraph -->
                <div>
                    <h5 style="margin: 0 0 6px 0; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; font-weight: 700;">
                        Strategic Rationale
                    </h5>
                    <p style="margin: 0; font-size: 13.5px; color: #334155; line-height: 1.6;">
                        ${escapeHtml(decision.reason || "Model hazard signals identify actionable risk factors eligible for mitigation.")}
                    </p>
                </div>

                <!-- ROI & Value Stats Strip -->
                ${selected ? `
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; padding-top: 14px; border-top: 1px solid #f1f5f9;">
                        <div style="border-right: 1px solid #f1f5f9; padding-right: 12px;">
                            <span style="display: block; font-size: 11px; color: #64748b; font-weight: 600; text-transform: uppercase;">Modeled Churn Delta</span>
                            <strong style="display: block; margin-top: 2px; font-size: 16px; color: #16a34a;">
                                -${safeNumber(roi.modeled_churn_improvement_pp, 0).toFixed(2)} pp
                            </strong>
                        </div>
                        <div style="border-right: 1px solid #f1f5f9; padding-right: 12px;">
                            <span style="display: block; font-size: 11px; color: #64748b; font-weight: 600; text-transform: uppercase;">Projected ROI</span>
                            <strong style="display: block; margin-top: 2px; font-size: 16px; color: #0f172a;">
                                ${safeNumber(roi.estimated_roi_pct, 0).toFixed(1)}%
                            </strong>
                        </div>
                        <div>
                            <span style="display: block; font-size: 11px; color: #64748b; font-weight: 600; text-transform: uppercase;">Estimated Net Value</span>
                            <strong style="display: block; margin-top: 2px; font-size: 16px; color: #0f172a;">
                                ${formatCurrency(roi.estimated_net_value)}
                            </strong>
                        </div>
                    </div>
                ` : ''}

            </div>

            <!-- Footer Disclaimer -->
            <div style="background: #f8fafc; border-top: 1px solid #f1f5f9; padding: 12px 24px; font-size: 11px; color: #94a3b8; font-style: italic;">
                ${escapeHtml(decision.decision_disclaimer || "Decision-support recommendation synthesized from Cox hazard regressions and counterfactual simulations. Observational findings do not guarantee causal treatment results.")}
            </div>

        </div>
    `;
}

/* ============================================================
    SIMULATION RESULT - BASIC SHELL
   ============================================================ */

function renderSimulationResult(simulated, originalCustomer, changes) {
    const container = document.getElementById("simulation-result");
    if (!container) return;

    const original = window.ChurnGuard.state.lastPrediction || state.lastPrediction;
    if (!original) return;

    // Convert decimal or percentage fields safely
    const originalChurnRaw = firstDefined(original.projected_churn, original.projected_churn_pct);
    const simulatedChurnRaw = firstDefined(simulated.projected_churn, simulated.projected_churn_pct);

    const originalChurn = Number(originalChurnRaw) <= 1 ? Number(originalChurnRaw) * 100 : Number(originalChurnRaw);
    const simulatedChurn = Number(simulatedChurnRaw) <= 1 ? Number(simulatedChurnRaw) * 100 : Number(simulatedChurnRaw);

    const churnChange = simulatedChurn - originalChurn;
    const retentionChange = -churnChange;

    const changeRows = Object.entries(changes)
        .map(([field, value]) => `
            <div class="simulation-change" style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #edf1f6; font-size: 12px;">
                <span style="color: #64748b;">${escapeHtml(field)}</span>
                <strong style="color: #1e293b;">${escapeHtml(value)}</strong>
            </div>
        `)
        .join("");

    const direction = churnChange < -0.01 ? "lower" : churnChange > 0.01 ? "higher" : "unchanged";

    container.innerHTML = `
        <div class="simulation-result-header" style="margin-bottom: 14px;">
            <span class="section-eyebrow" style="font-size: 10px; font-weight: 800; color: #1f6feb; letter-spacing: 0.1em; text-transform: uppercase;">
                SIMULATION RESULT
            </span>
            <h3 style="font-size: 16px; color: #0f172a; margin-top: 2px;">
                Model-based scenario comparison
            </h3>
        </div>

        <div class="simulation-metrics" style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-bottom: 16px;">
            <div class="sim-box" style="padding: 12px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;">
                <span style="font-size: 11px; color: #64748b; text-transform: uppercase; font-weight: 700; display: block;">Current Churn</span>
                <strong style="font-size: 18px; color: #0f172a;">${originalChurn.toFixed(1)}%</strong>
            </div>

            <div class="sim-box" style="padding: 12px; background: #ffffff; border: 1px solid #bfdbfe; border-radius: 8px;">
                <span style="font-size: 11px; color: #1f6feb; text-transform: uppercase; font-weight: 700; display: block;">Simulated Churn</span>
                <strong style="font-size: 18px; color: #1f6feb;">${simulatedChurn.toFixed(1)}%</strong>
            </div>

            <div class="sim-box" style="padding: 12px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;">
                <span style="font-size: 11px; color: #64748b; text-transform: uppercase; font-weight: 700; display: block;">Churn Change</span>
                <strong style="font-size: 15px; color: ${churnChange <= 0 ? '#16a34a' : '#dc2626'};">
                    ${churnChange > 0 ? "+" : ""}${churnChange.toFixed(2)} pp
                </strong>
            </div>

            <div class="sim-box" style="padding: 12px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;">
                <span style="font-size: 11px; color: #64748b; text-transform: uppercase; font-weight: 700; display: block;">Retention Change</span>
                <strong style="font-size: 15px; color: ${retentionChange >= 0 ? '#16a34a' : '#dc2626'};">
                    ${retentionChange > 0 ? "+" : ""}${retentionChange.toFixed(2)} pp
                </strong>
            </div>
        </div>

        <div class="simulation-changes" style="margin-bottom: 14px; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px;">
            <h4 style="font-size: 12px; text-transform: uppercase; color: #475569; margin-bottom: 6px;">
                Applied Scenario Changes
            </h4>
            ${changeRows}
        </div>

        <div class="simulation-interpretation" style="margin-bottom: 12px;">
            <p style="font-size: 12px; color: #334155; margin: 0;">
                The simulated configuration produces a <strong>${direction}</strong> modeled churn probability compared with the current baseline.
            </p>
        </div>

        <div class="simulation-disclaimer" style="font-size: 10px; color: #94a3b8; font-style: italic; border-top: 1px solid #edf1f6; padding-top: 8px;">
            Model-based counterfactual simulation, not a causal treatment effect or guaranteed customer outcome.
        </div>
    `;
}

/* ============================================================
   CLEAR RESULTS
   ============================================================ */

function clearPredictionResults() {
    state.lastPrediction = null;

    // Reset Overview Metrics
    setText("#projected-churn", "—");
    setText("#projected-retention", "—");
    setText("#hazard-multiplier", "—");
    setText("#risk-action", "Run an analysis to calculate risk.");

    const riskElement = $("#risk-tier");
    if (riskElement) {
        riskElement.textContent = "—";
        riskElement.className = "risk-badge risk-neutral";
    }

    // Reset Forecast Summary & Chart
    setText("#forecast-current-tenure", "—");
    setText("#forecast-horizon", "—");
    setText("#forecast-target-month", "—");
    setText("#forecast-target-retention", "—");

    const chart = $("#forecast-chart");
    if (chart) {
        chart.className = "chart-placeholder";
        chart.innerHTML = `
            <div class="placeholder-icon">↗</div>
            <h4>Forecast visualization</h4>
            <p>Run a customer analysis to populate the survival forecast.</p>
        `;
    }

    // Reset Explainability Drivers
    renderEmptyDriverState($("#risk-drivers"), "Prediction drivers will appear here.");
    renderEmptyDriverState($("#protective-drivers"), "Protective signals will appear here.");

    // Reset Interventions Container (Step 7)
    const interventionsContainer = document.getElementById("interventions-content");
    if (interventionsContainer) {
        interventionsContainer.innerHTML = `
            <div class="empty-state">
                <h3>No interventions yet</h3>
                <p>Run a prediction to generate retention actions.</p>
            </div>
        `;
    }

    // Reset Economics Metrics (Step 6)
    setText("#customer-value", "—");
    setText("#value-preserved", "—");
    setText("#net-value", "—");
    setText("#roi-percent", "—");
    setText("#roi-recommendation", "—");

    const roiIntervention = document.getElementById("roi-intervention");
    if (roiIntervention) {
        roiIntervention.innerHTML = `
            <div class="empty-state">
                Run a prediction to calculate intervention economics.
            </div>
        `;
    }

    // Reset Decision Panel (Step 8)
    const decisionContainer = document.getElementById("decision-content");
    if (decisionContainer) {
        decisionContainer.innerHTML = `
            <div class="empty-state">
                <h3>No decision generated</h3>
                <p>Run a customer prediction to generate retention decision intelligence.</p>
            </div>
        `;
    }

    // Reset What-If Simulation
    resetWhatIfSimulation();
}

/* ============================================================
   ALERTS
   ============================================================ */

function showAlert(message) {

    globalAlertMessage.textContent =
        message;

    globalAlert.classList.remove(
        "hidden"
    );

}


function hideAlert() {

    globalAlert.classList.add(
        "hidden"
    );

    globalAlertMessage.textContent =
        "";

}


/* ============================================================
   TEXT HELPERS
   ============================================================ */

function setText(selector, value) {

    const element =
        $(selector);


    if (!element) {
        return;
    }


    element.textContent =
        value ?? "—";

}


/* ============================================================
   FORMATTING
   ============================================================ */

function formatPercent(value) {

    if (
        value === null ||
        value === undefined ||
        Number.isNaN(Number(value))
    ) {
        return "—";
    }

    const num = Number(value);
    // If value is a decimal (e.g. 0.3749), convert to percentage
    const displayNum = num <= 1 && num > 0 ? num * 100 : num;

    return `${displayNum.toFixed(1)}%`;
}

function formatMultiplier(value) {

    if (
        value === null ||
        value === undefined ||
        Number.isNaN(Number(value))
    ) {
        return "—";
    }


    return `${Number(value).toFixed(2)}×`;

}


function formatCurrency(value) {

    if (
        value === null ||
        value === undefined ||
        Number.isNaN(Number(value))
    ) {
        return "—";
    }


    return new Intl.NumberFormat(
        "en-US",
        {
            style: "currency",
            currency: "USD",
            maximumFractionDigits: 0
        }
    ).format(
        Number(value)
    );

}


function normalizeRiskTier(value) {

    if (!value) {
        return "";
    }


    return String(value)
        .trim()
        .toUpperCase();

}


function normalizeProbability(value) {

    const number =
        Number(value);


    if (
        Number.isNaN(number)
    ) {
        return 0;
    }


    /*
        Accept either:

        0.85  -> 85%
        85    -> 85%
    */

    if (number > 1) {
        return number / 100;
    }


    return number;

}


/* ============================================================
   VALUE HELPER
   ============================================================ */

function firstDefined(...values) {

    for (const value of values) {

        if (
            value !== undefined &&
            value !== null
        ) {
            return value;
        }

    }


    return undefined;

}


/* ============================================================
   GLOBAL DEBUG ACCESS
   ============================================================ */

window.ChurnGuard = {

    state,

    runPrediction,

    checkApiHealth,

    clearPredictionResults,

    getApiBaseUrl: () =>
        API_BASE_URL

};

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function safeNumber(value, fallback = 0) {
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
}

async function updateTelemetryBar() {
  try {
    const res = await fetch('/metrics');
    if (!res.ok) return;
    const data = await res.json();
    const m = data.metrics;

    document.getElementById('tel-env').textContent = data.environment.toUpperCase();
    document.getElementById('tel-uptime').textContent = Math.round(m.uptime_seconds) + 's';
    document.getElementById('tel-predictions').textContent = m.predictions_total || 0;
    document.getElementById('tel-p95').textContent = (m.p95_latency_ms || 0).toFixed(1) + ' ms';

    const successPct = m.requests_total > 0 
      ? ((m.requests_successful / m.requests_total) * 100).toFixed(1) 
      : '100';
    document.getElementById('tel-success').textContent = successPct + '%';
  } catch (err) {
    console.debug('Telemetry offline:', err);
  }
}

// Fetch on load and refresh every 5 seconds
updateTelemetryBar();
setInterval(updateTelemetryBar, 5000);

/* ============================================================
   GLOBAL THEME TOGGLE CONTROLLER
   ============================================================ */

function initializeTheme() {
    const themeBtn = document.getElementById("theme-toggle");
    const themeIcon = document.getElementById("theme-icon");
    const themeLabel = document.getElementById("theme-label");

    const savedTheme = localStorage.getItem("churnguard_theme") || "light";
    applyTheme(savedTheme);

    if (themeBtn) {
        themeBtn.addEventListener("click", () => {
            const currentTheme = document.documentElement.getAttribute("data-theme") || "light";
            const nextTheme = currentTheme === "dark" ? "light" : "dark";
            applyTheme(nextTheme);
        });
    }

    function applyTheme(theme) {
        document.documentElement.setAttribute("data-theme", theme);
        localStorage.setItem("churnguard_theme", theme);

        if (themeIcon && themeLabel) {
            if (theme === "dark") {
                themeIcon.textContent = "☀️";
                themeLabel.textContent = "Light Mode";
            } else {
                themeIcon.textContent = "🌙";
                themeLabel.textContent = "Dark Mode";
            }
        }
    }
}

document.addEventListener("DOMContentLoaded", initializeTheme);