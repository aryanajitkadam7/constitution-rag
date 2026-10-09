FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY data/ data/
COPY reports/retrieval_results.json reports/retrieval_results.json
ENV PYTHONPATH=/app/src
# download the embedding model and build the index cache at build time
RUN python -c "from crag.pipeline import build; build()"
EXPOSE 8000
CMD ["uvicorn", "crag.api:app", "--host", "0.0.0.0", "--port", "8000"]
