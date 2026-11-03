import type { Document } from './api'

// A title search is a discovery aid, not a verified DOI or publisher link.
export function paperSearchUrl(title: string): string {
  return `https://scholar.google.com/scholar?q=${encodeURIComponent(`"${title}"`)}`
}
export function paperReference(doc: Document): string {
  if (doc.source === 'OpenAlex') {
    const authors = doc.authors?.length ? `${doc.authors.slice(0, 6).join(', ')}${doc.authors.length > 6 ? ', et al.' : ''}` : 'Authors unavailable'
    return `${authors} (${doc.publication_year ?? 'year unavailable'}). ${doc.title}${doc.venue ? `. ${doc.venue}` : ''}.
${doc.doi ?? doc.source_url ?? doc.catalog_url ?? paperSearchUrl(doc.title)}
Metadata from OpenAlex. Verify publication details and read the original paper before citing.`
  }
  return `${doc.title}\nSciFact / S2ORC abstract · document ${doc.doc_id}\nFind original: ${paperSearchUrl(doc.title)}\nVerify authors, year and publication details on the original paper before citing.`
}
