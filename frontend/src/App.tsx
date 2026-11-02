import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { health, search } from './api'
import type { Document, Mode, SearchResult } from './api'

type View = { status: 'idle' } | { status: 'loading' } | { status: 'error'; message: string } |
  { status: 'success'; result: SearchResult; mode: Mode; elapsed: number }
const suggestions = ['How does hybrid retrieval work?', 'What is the difference between precision and recall?', 'Who wrote the BM25 paper?']
function DocumentCard({ doc, index }: { doc: Document; index: number }) {
  const [expanded, setExpanded] = useState(false)
  return <article className="document">
    <div className="document-head"><span className="rank">{String(index + 1).padStart(2, '0')}</span><div><span className="eyebrow">{doc.category || 'DOCUMENT'} · {doc.doc_id}</span><h3>{doc.title}</h3></div><span className="score" title="Normalized blended retrieval score, not probability of correctness">{doc.hybrid_score?.toFixed(3) ?? '—'}<small>hybrid score</small></span></div>
    <p className="passage">{expanded || doc.text.length <= 360 ? doc.text : `${doc.text.slice(0, 360)}…`}</p>
    {doc.text.length > 360 && <button className="text-button" onClick={() => setExpanded(!expanded)}>{expanded ? 'Show less' : 'Read full passage'} <span aria-hidden="true">↗</span></button>}
    <div className="document-foot"><span>BM25 <b>{doc.bm25_score?.toFixed(2) ?? '—'}</b></span><span>Cosine similarity <b>{doc.semantic_score?.toFixed(3) ?? '—'}</b></span>{doc.source && <span className="source-name">{doc.source}</span>}</div>
  </article>
}
export default function App() {
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState<Mode>('search')
  const [alpha, setAlpha] = useState(0.5)
  const [topK, setTopK] = useState(5)
  const [view, setView] = useState<View>({ status: 'idle' })
  const [connection, setConnection] = useState<'checking' | 'connected' | 'offline'>('checking')
  const active = useRef<AbortController | null>(null)
  useEffect(() => {
    const controller = new AbortController()
    health(controller.signal).then(() => setConnection('connected')).catch(() => {
      if (!controller.signal.aborted) setConnection('offline')
    })
    return () => { controller.abort(); active.current?.abort() }
  }, [])
  async function run(value: string) {
    const question = value.trim()
    if (!question) return
    active.current?.abort()
    const controller = new AbortController()
    active.current = controller
    setQuery(question); setView({ status: 'loading' })
    const started = performance.now()
    try {
      const result = await search(mode, { query: question, alpha, top_k: topK }, controller.signal)
      if (!controller.signal.aborted) { setConnection('connected'); setView({ status: 'success', result, mode, elapsed: performance.now() - started }) }
    } catch (error) {
      if (!controller.signal.aborted) setView({ status: 'error', message: error instanceof Error ? error.message : 'Something went wrong. Please try again.' })
    }
  }
  function submit(event: FormEvent<HTMLFormElement>) { event.preventDefault(); void run(query) }
  const busy = view.status === 'loading'
  const result = view.status === 'success' ? view.result : undefined
  const excerpt = result?.outcome === 'cited excerpt' || result?.trace.some(event => event.mode === 'offline excerpt')
  return <div className="app">
    <header className="header"><a className="brand" href="#"><span className="brand-mark" aria-hidden="true">h<span>↗</span></span><span>hybrid<span className="brand-light"> / search</span></span></a><div className={`connection ${connection}`}><i aria-hidden="true" />{connection === 'connected' ? 'API connected' : connection === 'checking' ? 'Checking API' : 'API offline'}</div></header>
    <main>
      <section className="intro"><p className="eyebrow">EVIDENCE EXPLORER <span>01 / RETRIEVAL LAB</span></p><h1>Find the signal.<br /><em>Follow the evidence.</em></h1><p className="intro-copy">Search beyond exact words. Bring keyword precision and semantic meaning into the same conversation.</p></section>
      <div className="workspace">
        <aside className="settings"><div className="section-label">01 <span>SEARCH CONTROLS</span></div>
          <fieldset><legend>Experience</legend><div className="modes">{(['search', 'guided'] as const).map(value => <button type="button" key={value} aria-pressed={mode === value} onClick={() => setMode(value)}>{value === 'search' ? 'Search' : 'Guided RAG'}</button>)}</div><p className="hint">{mode === 'search' ? 'Explore passages ranked by combined relevance.' : 'Check evidence, retry once if needed, then answer or abstain.'}</p></fieldset>
          <fieldset><legend>Retrieval balance <span className="value">{Math.round(alpha * 100)} / {Math.round((1 - alpha) * 100)}</span></legend><label className="sr-only" htmlFor="balance">Keyword weight</label><input id="balance" type="range" min="0" max="1" step="0.05" value={alpha} onChange={e => setAlpha(Number(e.target.value))} /><div className="range-labels"><span>Keyword <b>{Math.round(alpha * 100)}%</b></span><span>Semantic <b>{Math.round((1 - alpha) * 100)}%</b></span></div></fieldset>
          <fieldset><legend>Evidence to retrieve</legend><label className="sr-only" htmlFor="top-k">Maximum documents</label><select id="top-k" value={topK} onChange={e => setTopK(Number(e.target.value))}>{[3, 5, 10, 20].map(n => <option key={n} value={n}>{n} documents</option>)}</select></fieldset>
          <div className="note"><span aria-hidden="true">↳</span><p><b>Grounded in your collection.</b> Scores measure retrieval relevance. They aren’t confidence scores for factual correctness.</p></div>
        </aside>
        <section className="explorer" aria-label="Search workspace"><form onSubmit={submit}><label className="section-label" htmlFor="query">02 <span>YOUR QUESTION</span></label><div className="query-box"><textarea id="query" placeholder="What would you like to understand?" maxLength={2000} rows={2} value={query} onChange={e => setQuery(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); if (!busy) void run(query) } }} /><div className="query-bottom"><span>Enter to search · Shift + Enter for a new line</span><button className="primary" type="submit" disabled={busy || !query.trim()}>{busy ? 'Searching…' : mode === 'search' ? 'Find evidence' : 'Ask with evidence'} <span aria-hidden="true">↗</span></button></div></div></form>
          <div className="suggestions"><span>TRY</span>{suggestions.map(s => <button key={s} disabled={busy} onClick={() => void run(s)}>{s} <span aria-hidden="true">↗</span></button>)}</div>
          <div className="results" aria-live="polite" aria-busy={busy}>
            {view.status === 'idle' && <div className="empty"><div className="empty-symbol" aria-hidden="true">⌕</div><p className="eyebrow">A GOOD QUESTION IS A STARTING POINT</p><h2>Your evidence belongs here.</h2><p>Ask a question or try an example above.<br />Explore the passages, their scores, and the path to an answer.</p><div className="empty-tags"><span>BM25</span><i>+</i><span>MiniLM</span><i>→</i><span>Relevant evidence</span></div></div>}
            {busy && <div className="loading"><span className="spinner" aria-hidden="true"/><h2>Following the evidence…</h2><p>{mode === 'guided' ? 'Retrieving passages and checking support. Local generation may take a moment.' : 'Ranking keyword and semantic matches.'}</p></div>}
            {view.status === 'error' && <div className="error" role="alert"><p className="eyebrow">SEARCH COULD NOT COMPLETE</p><h2>Let’s try that again.</h2><p>{view.message}</p><button className="text-button" onClick={() => void run(query)}>Retry search ↗</button></div>}
            {view.status === 'success' && <><div className="result-heading"><h2>{view.mode === 'guided' ? 'Evidence & answer' : 'Retrieved evidence'}</h2><span>{view.result.documents.length} passages · {(view.elapsed / 1000).toFixed(2)}s</span></div><p className="searched-query">For “{view.result.query}”</p>
              {view.result.answer && <article className="answer"><div className="answer-label"><span>{excerpt ? 'RETRIEVED EXCERPT' : view.result.outcome === 'insufficient evidence' ? 'INSUFFICIENT EVIDENCE' : view.result.outcome === 'blocked' ? 'REQUEST BLOCKED' : 'GUIDED ANSWER'}</span><span>{view.result.outcome}</span></div><p>{view.result.answer}</p>{excerpt && <small>Extractive fallback: this response uses retrieved text rather than a generated model answer.</small>}</article>}
              {view.result.trace.length > 0 && <details className="trace"><summary>Follow the workflow <span>{view.result.trace.length} steps</span></summary><ol>{view.result.trace.map((event, i) => <li key={`${event.step}-${i}`}><b>{event.step}{event.attempt ? ` · ${event.attempt}` : ''}</b><span>{event.result ?? event.mode ?? event.query ?? 'Completed'}</span>{event.reason && <small>{event.reason}</small>}</li>)}</ol></details>}
              {view.result.documents.length === 0 && <p className="no-results">No supporting passages were returned. Try a more specific question about your collection.</p>}
              {view.result.documents.map((doc, i) => <DocumentCard key={`${view.result.query}-${doc.doc_id}-${i}`} doc={doc} index={i} />)}
            </>}
          </div>
        </section>
      </div>
      <footer><span>HYBRID SEARCH / EVIDENCE EXPLORER</span><span>Built with React + TypeScript · Powered by FastAPI</span></footer>
    </main>
  </div>
}
