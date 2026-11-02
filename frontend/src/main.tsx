import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './styles.css'

if (import.meta.env.MODE === 'streamlit') {
  const { initializeStreamlit } = await import('./streamlit')
  initializeStreamlit()
}

ReactDOM.createRoot(document.getElementById('root')!).render(<React.StrictMode><App /></React.StrictMode>)
