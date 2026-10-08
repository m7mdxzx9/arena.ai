// @vitest-environment jsdom
import { describe, expect, it } from 'vitest'
import { ApiError, formatApiError, localizedApiError } from '../src/api'

describe('formatApiError', () => {
  it('recognizes application errors and primitive failures', () => {
    expect(formatApiError(new ApiError('Dataset is too large.', 413, 'file_too_large'))).toBe('Dataset is too large.')
    expect(formatApiError({ error: 'Invalid target', code: 'bad_target' })).toBe('Invalid target')
    expect(formatApiError({ detail: [{ msg: 'Field is required' }, { msg: 'Must be text' }] })).toBe('Field is required; Must be text')
    expect(formatApiError('offline')).toBe('offline')
    expect(formatApiError(503)).toBe('503')
  })

  it('localizes structured backend error codes and retains safe legacy messages', () => {
    const translate = (key: string) => key === 'errors.file_too_large' ? 'الملف أكبر من الحد.' : key === 'errors.application_error' ? 'حدث خطأ.' : key
    expect(localizedApiError(new ApiError('Too large', 413, 'file_too_large'), translate)).toBe('الملف أكبر من الحد.')
    expect(localizedApiError(new Error('Legacy detail'), translate)).toBe('Legacy detail')
    expect(localizedApiError({ code: 'unknown_code', message: '' }, translate)).toBe('حدث خطأ.')
  })

  it('never throws for malformed objects or throwing proxies', () => {
    const dangerous = new Proxy({}, { get() { throw new Error('getter exploded') } })
    expect(() => formatApiError(dangerous)).not.toThrow()
    expect(formatApiError(dangerous)).toBe('Something went wrong. Please try again.')
    expect(formatApiError(null)).toBe('Something went wrong. Please try again.')
  })
})
