// @vitest-environment jsdom
import { describe, expect, it } from 'vitest'
import { ApiError, formatApiError } from '../src/api'

describe('formatApiError', () => {
  it('recognizes application errors and primitive failures', () => {
    expect(formatApiError(new ApiError('Dataset is too large.', 413, 'file_too_large'))).toBe('Dataset is too large.')
    expect(formatApiError({ error: 'Invalid target', code: 'bad_target' })).toBe('Invalid target')
    expect(formatApiError({ detail: [{ msg: 'Field is required' }, { msg: 'Must be text' }] })).toBe('Field is required; Must be text')
    expect(formatApiError('offline')).toBe('offline')
    expect(formatApiError(503)).toBe('503')
  })

  it('never throws for malformed objects or throwing proxies', () => {
    const dangerous = new Proxy({}, { get() { throw new Error('getter exploded') } })
    expect(() => formatApiError(dangerous)).not.toThrow()
    expect(formatApiError(dangerous)).toBe('Something went wrong. Please try again.')
    expect(formatApiError(null)).toBe('Something went wrong. Please try again.')
  })
})
