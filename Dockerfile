FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "entailforge.api:app", "--host", "0.0.0.0", "--port", "8000"]
