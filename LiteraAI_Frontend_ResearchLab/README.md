# LiteraAI Frontend

Frontend-only React + Vite interface for the RAG Doctor research workflow.

## Run
```bash
npm install
cp .env.example .env
npm run dev
```

## Existing backend integration
Set `VITE_API_URL` and update the endpoint paths in `.env` to match your existing RAG Doctor FastAPI routes. This frontend does not include or create a backend.

Expected requests:
- Ingest: multipart form field `files` (PDF upload)
- Ask: JSON `{ "question": "...", "top_k": 3 }`
- Analyze: JSON `{ "document_id": "..." }`
- Research search: multipart form fields `research_query`, optional `document`, `limit`, `year_from`, `year_to`, `open_access_only`, `sort_by`

The UI displays a demo mode when no API URL is configured. Demo responses are clearly labelled and are not real model output.

Keep API credentials in your existing backend environment. Never put OpenAlex keys in Vite environment variables.
