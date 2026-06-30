# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Builder
# Install dependencies in a separate stage so only the final, clean image
# is shipped — not the pip cache or any build-time tools.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Upgrade pip to avoid legacy-resolver issues
RUN pip install --upgrade pip

# Copy only requirements first — changes here trigger a cache-layer rebuild
# but code changes alone do not.
COPY requirements.txt .

# Install into an isolated prefix so the runtime stage can copy them cleanly
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Runtime
# Minimal image that only contains the app code and installed packages.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Create a non-root user for security — never run as root in production
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /install /usr/local

# Copy application source code
COPY . .

# Ensure Python can find the application modules
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Drop privileges — run as non-root
USER appuser

# Expose the default application port
EXPOSE 8000

# Healthcheck so Docker knows when the container is ready.
# Render performs its own HTTP health check on / — this is a belt-and-braces
# check for local `docker run` usage.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python3 -c \
        "import urllib.request, os; urllib.request.urlopen('http://localhost:' + os.environ.get('PORT','8000') + '/')" \
        || exit 1

# Use exec inside sh so SIGTERM is delivered directly to uvicorn, enabling
# graceful shutdown on Render (shell-form CMD swallows signals).
# $PORT is injected by Render at runtime; defaults to 8000 locally.
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 75"]
