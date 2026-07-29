# Surfer Agent

Open-source agentic AI assistant for exploring, retrieving, and visualizing
oceanographic datasets via natural language. Works with THREDDS and ERDDAP
data servers in one unified dashboard.

## Features

- Search, describe, and summarize servers and datasets from a URL
- Filter datasets by variable, geographic bounds, and time; get a download URL
- In-browser graphs and maps, exportable to PNG/HTML

## Installation

### 1. Install Docker

- **Mac/Windows**: [Docker Desktop](https://www.docker.com/products/docker-desktop/), then start it
- **Linux**: [Docker Engine](https://docs.docker.com/engine/install/) (includes the `docker compose` plugin on recent installs)

```bash
docker --version
docker compose version
```

### 2. Get an OpenRouter API key

1. Sign up at [openrouter.ai](https://openrouter.ai/)
2. Create a key at [openrouter.ai/keys](https://openrouter.ai/keys)

Free-tier models are available, so no payment method is required to try it
— filter by `max_price=0` on [openrouter.ai/models](https://openrouter.ai/models).

### 3. Get the code

```bash
git clone https://github.com/Dyza-Labs/surfer.git
cd surfer
```

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```bash
OPENROUTER_API_KEY=<your key from step 2>
ORCHESTRATOR_MODEL=<model id, e.g. nvidia/nemotron-nano-9b-v2:free>
SUBAGENT_MODEL=<model id, e.g. nvidia/nemotron-3-ultra-550b-a55b:free>
POSTGRES_URI=postgresql://surfer:surfer@postgres:5432/surfer
```

- Model ids: browse [openrouter.ai/models](https://openrouter.ai/models) (the examples above are free)
- `POSTGRES_URI`'s host must stay `postgres` (the container's service name), not `localhost`
- Leave `LANGSMITH_*` blank unless you want tracing (separate account)

### 5. Start everything

```bash
docker compose up --build
```

First run takes a few minutes; later runs are fast.

### 6. Open it

Once the logs show Streamlit's "You can now view your app": http://localhost:8501

## Managing the deployment

```bash
docker compose down     # stop, keep chat history
docker compose down -v  # stop, wipe chat history
docker compose up       # restart, no rebuild
```

**Hosted DB instead of local**: the bundled Postgres is tied to this
machine's disk. For durable/hosted storage, sign up at
[Neon](https://neon.tech) (free tier), create a database, and swap
`POSTGRES_URI` for the connection string it gives you.

## Troubleshooting

- **Port conflict on startup** — something else is using 8501/5432; stop it or remap the port in `docker-compose.yml`
- **App loads but every request errors** — check `OPENROUTER_API_KEY` is correct and your account can access the models you set

## Development Quickstart

```bash
uv sync --dev        # install deps
uv run langgraph dev # run locally
```

Integration tests are skipped unless `OPENROUTER_API_KEY` is set.
