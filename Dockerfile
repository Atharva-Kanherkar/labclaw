FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY labclaw ./labclaw
COPY samples ./samples
COPY tests/fixtures ./tests/fixtures

RUN pip install --no-cache-dir .

ENV LABCLAW_DATA_DIR=/data
ENV LABCLAW_FIXTURE_MODE=1
ENV LABCLAW_CORS_ORIGINS=*

EXPOSE 8000

CMD ["sh", "-c", "uvicorn labclaw.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
