FROM python:3.12-slim

# cartopy needs build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ gcc libgeos-dev libproj-dev proj-data proj-bin \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY . .

RUN uv sync --no-dev

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app.py", "--server.address=0.0.0.0"]
