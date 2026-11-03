import { describe, expect, it } from 'vitest'
import { paperReference, paperSearchUrl } from './research'

describe('paper discovery references', () => {
  it('encodes the entire title as a search, including URL metacharacters', () => {
    const title = 'Memory & exercise? A #1 study'
    const url = new URL(paperSearchUrl(title))
    expect(url.origin).toBe('https://scholar.google.com')
    expect(url.searchParams.get('q')).toBe(`"${title}"`)
    expect(url.hash).toBe('')
  })
  it('does not invent bibliographic metadata', () => {
    const reference = paperReference({ doc_id:'123', title:'Exercise and memory', text:'Abstract' })
    expect(reference).toContain('document 123')
    expect(reference).toContain('Verify authors, year and publication details')
    expect(reference).not.toContain('doi.org')
  })
})

describe('catalog publication metadata', () => {
  it('uses the supplied authors, year, venue and DOI without claiming a citation style', () => {
    const reference = paperReference({ doc_id:'W1', source:'OpenAlex', title:'Attention research', text:'Abstract', authors:['A. Researcher'], publication_year:2024, venue:'Journal', doi:'https://doi.org/10.1234/example' })
    expect(reference).toContain('A. Researcher (2024). Attention research. Journal.')
    expect(reference).toContain('https://doi.org/10.1234/example')
    expect(reference).toContain('Verify publication details')
  })
})
