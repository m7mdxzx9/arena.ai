import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'
import './rtl-and-labs.css'
import App from './App.tsx'
import { I18nProvider } from './i18n.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <I18nProvider>
      <App />
    </I18nProvider>
  </StrictMode>,
)
