FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY run.py .
COPY static_tools_dict.py .

RUN useradd --create-home --uid 1000 maestro && chown -R maestro:maestro /app
USER maestro

EXPOSE 8000

# The CMD is removed from here as Smithery.ai will use the 
# startCommand from smithery.yaml to run the container. 