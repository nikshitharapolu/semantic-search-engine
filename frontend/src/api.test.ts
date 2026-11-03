import { afterEach, describe, expect, it, vi } from 'vitest'
import { search } from './api'
const options = { query: 'hybrid search', alpha: 0.5, top_k: 3 }
const signal = () => new AbortController().signal
function respond(body: unknown, status = 200) {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })))
}
afterEach(() => vi.unstubAllGlobals())
describe('typed API boundary', () => {
  it('maps search results and handles null optional scores', async () => {
    respond({ query: options.query, results: [{ doc_id: 'D1', text: 'Evidence', title: 'Title', hybrid_score: null }] })
    const result = await search('search', options, signal())
    expect(result.documents[0].hybrid_score).toBeUndefined()
    expect(fetch).toHaveBeenCalledWith('/api/research/search', expect.objectContaining({ body: JSON.stringify(options) }))
  })
  it('maps guided evidence and trace to the shared view model', async () => {
    respond({ query: options.query, answer: 'Evidence [1]', sources: [{ doc_id: 'D1', text: 'Evidence' }], outcome: 'cited excerpt', trace: [{ step: 'generate', mode: 'offline excerpt' }] })
    const result = await search('guided', options, signal())
    expect(result.trace[0].mode).toBe('offline excerpt')
    expect(result.documents[0].title).toBe('Untitled document')
    expect(fetch).toHaveBeenCalledWith('/api/research/ask/guided', expect.anything())
  })
  it('accepts an abstention with no sources', async () => {
    respond({ query: options.query, answer: 'Insufficient evidence.', sources: [], outcome: 'insufficient evidence' })
    expect((await search('guided', options, signal())).documents).toEqual([])
  })
  it('keeps closest results separate from supporting sources', async () => {
    respond({ query: options.query, answer: 'Not enough evidence.', sources: [], outcome: 'insufficient evidence', related_results: [{ doc_id: 123, text: 'Nearby topic', category: 'SciFact' }], query_suggestion: 'corrected query' })
    const result = await search('guided', options, signal())
    expect(result.documents).toEqual([])
    expect(result.relatedDocuments[0].doc_id).toBe('123')
    expect(result.querySuggestion).toBe('corrected query')
  })
  it('keeps catalog metadata and drops unsafe links', async () => {
    respond({ query: options.query, results: [{ doc_id:'W1', title:'Paper', text:'Abstract', authors:['Author',null], publication_year:2024, source_url:'javascript:alert(1)', doi:'https://doi.org/10.1234/example' }], collection:{name:'OpenAlex',description:'Candidate ranking'} })
    const result = await search('search', {...options, subject:'computer-science'}, signal())
    expect(result.documents[0].authors).toEqual(['Author'])
    expect(result.documents[0].source_url).toBeUndefined()
    expect(result.documents[0].doi).toBe('https://doi.org/10.1234/example')
    expect(result.collection?.name).toBe('OpenAlex')
  })
  it('rejects malformed successful responses', async () => {
    respond({ query: options.query, results: [{ doc_id: 'D1' }] })
    await expect(search('search', options, signal())).rejects.toThrow('invalid document')
  })
  it('surfaces backend error messages', async () => {
    respond({ detail: 'Service unavailable.' }, 503)
    await expect(search('search', options, signal())).rejects.toThrow('Service unavailable.')
  })
  it('explains a development proxy connection failure', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('', { status: 500 })))
    await expect(search('search', options, signal())).rejects.toThrow('FastAPI is running on port 8000')
  })
  it('preserves cancellation without mislabeling it as a network failure', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new DOMException('Canceled', 'AbortError')))
    await expect(search('search', options, signal())).rejects.toMatchObject({ name: 'AbortError' })
  })
  it('gives an actionable connection error', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')))
    await expect(search('search', options, signal())).rejects.toThrow('port 8000')
  })
})
