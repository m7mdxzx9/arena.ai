/* oxlint-disable react/only-export-components -- the provider and its typed hook share one localization boundary */
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import en from './locales/en/common'
import ar from './locales/ar/common'

export type Language = 'en' | 'ar'
type Params = Record<string, string | number>
type I18nValue = string | Record<string, unknown>

type I18nContextValue = {
  language: Language
  direction: 'ltr' | 'rtl'
  setLanguage: (language: Language) => void
  t: (key: string, params?: Params) => string
  locale: string
}

const dictionaries: Record<Language, Record<string, unknown>> = { en, ar }
const LANGUAGE_KEY = 'neural-forge-language'

function storedLanguage(): Language {
  try {
    return localStorage.getItem(LANGUAGE_KEY) === 'ar' ? 'ar' : 'en'
  } catch {
    return 'en'
  }
}

function resolve(dictionary: Record<string, unknown>, key: string): I18nValue | undefined {
  let value: unknown = dictionary
  for (const part of key.split('.')) {
    if (!value || typeof value !== 'object' || !(part in value)) return undefined
    value = (value as Record<string, unknown>)[part]
  }
  return value as I18nValue
}

function interpolate(template: string, params: Params): string {
  return template.replace(/\{\{(\w+)\}\}/g, (_, name: string) => (name in params ? String(params[name]) : `{{${name}}}`))
}

const fallbackT = (key: string, params: Params = {}) => {
  const value = resolve(dictionaries.en, key)
  return interpolate(typeof value === 'string' ? value : key, params)
}

// The English default keeps isolated component tests usable; the production root
// always supplies I18nProvider.
export const I18nContext = createContext<I18nContextValue>({
  language: 'en', direction: 'ltr', setLanguage: () => {}, t: fallbackT, locale: 'en',
})

export function I18nProvider({ children }: { children: ReactNode }) {
  const [language, setLanguageState] = useState<Language>(storedLanguage)
  const setLanguage = useCallback((next: Language) => {
    const safe = next === 'ar' ? 'ar' : 'en'
    setLanguageState(safe)
    try { localStorage.setItem(LANGUAGE_KEY, safe) } catch { /* storage may be unavailable */ }
  }, [])
  const t = useCallback((key: string, params: Params = {}) => {
    const translated = resolve(dictionaries[language], key)
    const fallback = resolve(dictionaries.en, key)
    const value = typeof translated === 'string' ? translated : typeof fallback === 'string' ? fallback : key
    return interpolate(value, params)
  }, [language])
  useEffect(() => {
    document.documentElement.lang = language
    document.documentElement.dir = language === 'ar' ? 'rtl' : 'ltr'
    document.body.classList.toggle('rtl', language === 'ar')
  }, [language])
  const value = useMemo<I18nContextValue>(() => ({
    language,
    direction: language === 'ar' ? 'rtl' : 'ltr',
    setLanguage,
    t,
    locale: language === 'ar' ? 'ar-SA' : 'en',
  }), [language, setLanguage, t])
  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>
}

export function useI18n(): I18nContextValue {
  return useContext(I18nContext)
}

/** Keep source code, commands, identifiers and equations left-to-right in RTL UI. */
export function Ltr({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <span dir="ltr" className={`technical-ltr ${className}`.trim()}>{children}</span>
}
