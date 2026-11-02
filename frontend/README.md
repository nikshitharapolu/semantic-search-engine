# Hybrid Search — React + TypeScript

A local frontend for the existing FastAPI backend. Uses strict TypeScript,
React state unions, runtime validation at the API boundary, cancellation of
superseded requests, and Vite's same-origin API proxy.

## Run

From the repository root, start the backend in one terminal:

```sh
source .venv/bin/activate
uvicorn api:app --app-dir src --host 127.0.0.1 --port 8000
```

In another terminal:

```sh
cd frontend
npm ci
npm run dev
```

Open http://127.0.0.1:5173. Enter runs a search; Shift+Enter adds a line.
Guided RAG uses the existing Ollama/extractive fallback behavior. The UI labels
extractive answers and shows evidence-check decisions. No API key is exposed
to the browser. The frontend does not load the repository's private `.env`.

An existing local Docker Compose API on port 8000 can also be used. To proxy
a different backend, set `API_PROXY_TARGET` before starting Vite. This is a
server-side variable; do not put private keys into `VITE_` variables.

## Checks

```sh
npm test
npm run build
```

The build includes strict type checking. `dist/` is a static build; deploying it
requires a server/reverse proxy that routes `/api/*` to FastAPI with `/api`
stripped. Vite's development proxy does not configure production hosting.
`npm run preview` previews static files only and does not proxy API requests.

This adds a React interface alongside Streamlit. It does not deploy anything,
start AWS resources, or enable LangSmith tracing automatically. Google Fonts
are optional; the stylesheet includes local serif and sans-serif fallbacks.
