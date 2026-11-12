# Build the model first (python -m bikeforecast.train) so models/model.joblib exists.
FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir pandas numpy requests holidays scikit-learn joblib fastapi "uvicorn[standard]"

COPY src ./src
COPY models ./models
RUN pip install --no-cache-dir --no-deps .

EXPOSE 8000
CMD ["uvicorn", "bikeforecast.api:app", "--host", "0.0.0.0", "--port", "8000"]
