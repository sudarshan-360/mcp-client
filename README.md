# Oncology Decision Support — FastAPI Backend

## Project layout

```
oncology_backend/
├── app/
│   ├── main.py                  # FastAPI app + lifespan
│   ├── config.py                # Settings (env vars)
│   ├── api/
│   │   └── routes/
│   │       ├── chat.py          # POST /chat/stream (SSE)
│   │       ├── tools.py         # GET  /tools
│   │       └── health.py        # GET  /health
│   ├── mcp/
│   │   ├── client.py            # Async MCP client wrapper
│   │   └── manager.py           # Connection lifecycle
│   ├── llm/
│   │   ├── groq_client.py       # Groq SDK wrapper
│   │   └── orchestrator.py      # Agentic tool-call loop
│   ├── models/
│   │   └── chat.py              # Pydantic I/O models
│   ├── core/
│   │   └── session.py           # Session management
│   └── db/
│       └── mongo.py             # MongoDB client
├── .env.example
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Quick start

```bash
cp .env.example .env           # fill in your keys
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Server + MCP status |
| GET | /tools | List all MCP tools |
| POST | /chat/stream | Streaming chat (SSE) |
| GET | /sessions/{id} | Fetch session history |
| DELETE | /sessions/{id} | Clear session |
