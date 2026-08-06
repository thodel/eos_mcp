FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY build_db.py db.py server.py ./

# hgb.db is mounted at runtime (see docker-compose.yml)
# Build it first with: docker run --rm -v /path/to/data:/data hgb-mcp python build_db.py --xml /data/hgb_full.xml --db /data/hgb.db

EXPOSE 8000

CMD ["python", "server.py", "--db", "/data/hgb.db", "--host", "0.0.0.0", "--port", "8000"]
