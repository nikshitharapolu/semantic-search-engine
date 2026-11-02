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

## Public Streamlit deployment

The public app at `src/app.py` hosts this interface as a bidirectional Streamlit
component. Requests use Python directly; a separate FastAPI server is not needed.
The standalone Vite development mode continues to use the FastAPI proxy.

Build the public assets before committing frontend changes:

```sh
npm ci
npm run build:streamlit
```

Commit `streamlit_build/` with the source. It is intentionally tracked because
Streamlit Community Cloud installs Python dependencies but does not run our Node
build. Never hand-edit generated assets. No frontend secrets are embedded.

From the repository root, preview the hosted version with:

```sh
streamlit run src/app.py
```

Community Cloud should track `main`, with `src/app.py` as its entry point. A push
to that branch triggers redeployment. This public mode uses the bundled corpus,
extractive answers and in-memory/session-local request handling. It never calls
Ollama, writes uploaded documents, starts AWS resources or enables LangSmith
tracing. Resource limits and sleep behavior of the free hosting still apply.
The prior Python-widget interface is retained at `src/legacy_app.py` for local use.

The hosted configuration disables Streamlit file watching to avoid scanning
Transformers lazy modules. Restart local Streamlit after Python edits; Community
Cloud still redeploys on GitHub pushes.
