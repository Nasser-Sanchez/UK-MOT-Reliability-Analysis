FROM python:3.11-slim

WORKDIR /app

# Install uv (fast pip replacement)
RUN pip install --no-cache-dir uv

# Install dependencies (uv caches automatically)
RUN uv pip install --system \
    streamlit \
    duckdb \
    duckdb-engine \
    pandas \
    pyarrow \
    plotly \
    seaborn \
    matplotlib

# Copy application code
COPY app/ ./app/

# Expose Streamlit port
EXPOSE 8501

# Environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_MAX_UPLOAD_SIZE=100

# Run the app
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
