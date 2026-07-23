FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY anamnesis/ ./anamnesis/
COPY knowledge_base/ ./knowledge_base/
COPY web/ ./web/
COPY run.py .

# The database and the vector index are built on first start, inside the container.
ENV ANAMNESIS_HOST=0.0.0.0
EXPOSE 8000

CMD ["python", "run.py"]
