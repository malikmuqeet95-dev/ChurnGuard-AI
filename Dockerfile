FROM python:3.12-slim

# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ------------------------------------------------------------
# Working directory
# ------------------------------------------------------------

WORKDIR /app

# ------------------------------------------------------------
# Install production dependencies
# ------------------------------------------------------------

COPY requirements-prod.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-prod.txt

# ------------------------------------------------------------
# Copy application
# ------------------------------------------------------------

COPY backend ./backend
COPY frontend ./frontend

# ------------------------------------------------------------
# Create non-root application user
# ------------------------------------------------------------

RUN useradd \
        --create-home \
        --shell /usr/sbin/nologin \
        appuser \
    && chown -R appuser:appuser /app

USER appuser

# ------------------------------------------------------------
# Runtime configuration
# ------------------------------------------------------------

ENV APP_ENV=production
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV LOG_LEVEL=INFO

# ------------------------------------------------------------
# Expose API
# ------------------------------------------------------------

EXPOSE 8000

# ------------------------------------------------------------
# Health check
# ------------------------------------------------------------

HEALTHCHECK --interval=30s \
            --timeout=5s \
            --start-period=30s \
            --retries=3 \
            CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

# ------------------------------------------------------------
# Production server
# ------------------------------------------------------------

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]