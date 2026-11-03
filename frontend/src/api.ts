export interface Document {
  doc_id: string
  title: string
  text: string
  category?: string
  source?: string
  authors?: string[]
  publication_year?: number
  venue?: string
  doi?: string
  source_url?: string
  catalog_url?: string
  bm25_score?: number
  semantic_score?: number
  hybrid_score?: number
}
export interface TraceEvent { step: string; result?: string; reason?: string; query?: string; mode?: string; attempt?: number }
export type Mode = 'search' | 'guided'
export type Subject = 'all' | 'computer-science' | 'engineering' | 'life-sciences' | 'health-sciences' | 'social-sciences' | 'physical-sciences'
export interface SearchOptions { query: string; alpha: number; top_k: number; subject?: Subject }
export interface SearchResult {
  query: string
  documents: Document[]
  answer?: string
  outcome?: string
  trace: TraceEvent[]
  relatedDocuments: Document[]
  querySuggestion?: string
  collection?: { name: string; description: string }
}
const object = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null && !Array.isArray(value)
function documents(value: unknown): Document[] {
  if (!Array.isArray(value)) throw new Error('The API returned an invalid document list.')
  return value.map((row: unknown) => {
    if (!object(row) || !['string', 'number'].includes(typeof row.doc_id) || typeof row.text !== 'string') {
      throw new Error('The API returned an invalid document.')
    }
    const doc: Document = { doc_id: String(row.doc_id), title: typeof row.title === 'string' ? row.title : 'Untitled document', text: row.text }
    for (const key of ['category', 'source', 'venue'] as const) if (typeof row[key] === 'string') doc[key] = row[key]
    if (Array.isArray(row.authors)) doc.authors = row.authors.filter((name): name is string => typeof name === 'string')
    if (typeof row.publication_year === 'number' && Number.isInteger(row.publication_year)) doc.publication_year = row.publication_year
    for (const key of ['doi', 'source_url', 'catalog_url'] as const) {
      if (typeof row[key] !== 'string') continue
      try { const url = new URL(row[key]); if (url.protocol === 'https:' && !url.username && !url.password) doc[key] = url.href } catch { /* Ignore malformed catalog URLs. */ }
    }
    for (const key of ['bm25_score', 'semantic_score', 'hybrid_score'] as const) {
      const score = row[key]
      if (typeof score === 'number' && Number.isFinite(score)) doc[key] = score
    }
    return doc
  })
}
async function request(path: string, init: RequestInit): Promise<unknown> {
  if (import.meta.env.MODE === 'streamlit') {
    const { streamlitRequest } = await import('./streamlit')
    return streamlitRequest(path, init)
  }
  let response: Response
  try { response = await fetch(`/api${path}`, init) }
  catch (error) {
    if (error instanceof Error && error.name === 'AbortError') throw error
    throw new Error('Cannot reach the search API. Check that the FastAPI backend is running on port 8000.')
  }
  const body: unknown = await response.json().catch(() => null)
  if (!response.ok) {
    if (body === null && response.status >= 500) throw new Error('The API proxy could not reach a working backend. Check that FastAPI is running on port 8000.')
    const detail = object(body) && typeof body.detail === 'string' ? body.detail : `Request failed (${response.status}). Please try again.`
    throw new Error(detail)
  }
  return body
}
export async function health(signal: AbortSignal): Promise<void> {
  const body = await request('/health', { signal })
  if (!object(body) || body.status !== 'healthy') throw new Error('API is not healthy.')
}
export async function search(mode: Mode, options: SearchOptions, signal: AbortSignal): Promise<SearchResult> {
  const prefix = import.meta.env.MODE === 'streamlit' ? '' : '/research'
  const body = await request(`${prefix}${mode === 'guided' ? '/ask/guided' : '/search'}`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(options), signal,
  })
  if (!object(body) || typeof body.query !== 'string') throw new Error('The API response is incomplete.')
  if (mode === 'guided' && typeof body.answer !== 'string') throw new Error('The API response is missing its answer.')
  const trace: TraceEvent[] = []
  if (Array.isArray(body.trace)) for (const event of body.trace) {
    if (!object(event) || typeof event.step !== 'string') continue
    const parsed: TraceEvent = { step: event.step }
    for (const key of ['result', 'reason', 'query', 'mode'] as const) if (typeof event[key] === 'string') parsed[key] = event[key]
    if (typeof event.attempt === 'number') parsed.attempt = event.attempt
    trace.push(parsed)
  }
  return { query: body.query, documents: documents(mode === 'guided' ? body.sources : body.results),
    answer: typeof body.answer === 'string' ? body.answer : undefined,
    outcome: typeof body.outcome === 'string' ? body.outcome : undefined, trace,
    relatedDocuments: Array.isArray(body.related_results) ? documents(body.related_results) : [],
    querySuggestion: typeof body.query_suggestion === 'string' ? body.query_suggestion : undefined,
    collection: object(body.collection) && typeof body.collection.name === 'string' && typeof body.collection.description === 'string'
      ? { name: body.collection.name, description: body.collection.description } : undefined }
}
