// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { I18nProvider, Ltr, useI18n } from '../src/i18n'
import en from '../src/locales/en/common'
import ar from '../src/locales/ar/common'

function keys(value: Record<string, unknown>, prefix = ''): string[] {
  return Object.entries(value).flatMap(([key, child]) => {
    const path = prefix ? `${prefix}.${key}` : key
    return child && typeof child === 'object' ? keys(child as Record<string, unknown>, path) : [path]
  }).sort()
}

function Probe() {
  const { t, language, setLanguage } = useI18n()
  return <div><span data-testid="language">{language}</span><span>{t('openLab.title')}</span><button onClick={() => setLanguage(language === 'ar' ? 'en' : 'ar')}>switch</button><Ltr>model.fit(X, y)</Ltr></div>
}

beforeEach(() => localStorage.clear())
afterEach(() => { cleanup(); document.documentElement.dir = 'ltr'; document.documentElement.lang = 'en' })

describe('localization resources and RTL', () => {
  it('keeps Arabic and English resource keys in parity', () => {
    expect(keys(ar as unknown as Record<string, unknown>)).toEqual(keys(en as unknown as Record<string, unknown>))
  })

  it('persists Arabic and applies RTL while technical content remains LTR', () => {
    localStorage.setItem('neural-forge-language', 'ar')
    render(<I18nProvider><Probe /></I18nProvider>)
    expect(screen.getByTestId('language').textContent).toBe('ar')
    expect(screen.getByText('المختبر المفتوح')).toBeTruthy()
    expect(document.documentElement.dir).toBe('rtl')
    expect(screen.getByText('model.fit(X, y)').getAttribute('dir')).toBe('ltr')
    fireEvent.click(screen.getByText('switch'))
    expect(document.documentElement.dir).toBe('ltr')
    expect(localStorage.getItem('neural-forge-language')).toBe('en')
  })
})
