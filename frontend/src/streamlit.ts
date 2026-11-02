import { Streamlit } from 'streamlit-component-lib'
import type { RenderData } from 'streamlit-component-lib'

type Pending = { resolve: (value: unknown) => void; reject: (error: Error) => void; cleanup: () => void }
const pending = new Map<string, Pending>()
let initialized = false
export function initializeStreamlit(): void {
  if (initialized) return
  initialized = true
  Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, (event: Event) => {
    const data = (event as CustomEvent<RenderData>).detail
    const response: unknown = data.args.response
    if (typeof response !== 'object' || response === null || !('id' in response) || typeof response.id !== 'string') return
    const item = pending.get(response.id)
    if (!item) return
    item.cleanup(); pending.delete(response.id)
    if ('error' in response && typeof response.error === 'string') item.reject(new Error(response.error))
    else item.resolve('body' in response ? response.body : null)
  })
  Streamlit.setComponentReady()
  const resize = () => Streamlit.setFrameHeight(Math.max(850, document.getElementById('root')?.scrollHeight ?? 850))
  new ResizeObserver(resize).observe(document.getElementById('root') ?? document.body)
  resize()
}
export async function streamlitRequest(path: string, init: RequestInit): Promise<unknown> {
  initializeStreamlit()
  if (path === '/health') return { status: 'healthy' }
  const id = crypto.randomUUID()
  const signal = init.signal
  if (signal?.aborted) throw new DOMException('Canceled', 'AbortError')
  return new Promise((resolve, reject) => {
    const abort = () => { pending.get(id)?.cleanup(); pending.delete(id); reject(new DOMException('Canceled', 'AbortError')) }
    const timer = window.setTimeout(() => {
      pending.get(id)?.cleanup(); pending.delete(id)
      reject(new Error('Search is taking longer than expected. The app may be warming up. Please try again.'))
    }, 180_000)
    const cleanup = () => { window.clearTimeout(timer); signal?.removeEventListener('abort', abort) }
    pending.set(id, { resolve, reject, cleanup })
    signal?.addEventListener('abort', abort, { once: true })
    Streamlit.setComponentValue({ id, path, payload: typeof init.body === 'string' ? JSON.parse(init.body) : {} })
  })
}
