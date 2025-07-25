import { useState } from 'react'
import reactLogo from './assets/react.svg'
import rfLogo from './assets/rflogo.png'
import viteLogo from '/vite.svg'
import './App.css'
import Header from './components/Header'
import Footer from './components/Footer'
import ReactLogo2 from './assets/react.svg?react'

function App() {
  const [count, setCount] = useState(0)
  const greeting = import.meta.env.VITE_GREETING

  return (
    <>
      <Header />
      <div>
        <a href="https://vite.dev" target="_blank">
          <img src={viteLogo} className="logo" alt="Vite logo" />
        </a>

        {/* <a href="https://react.dev" target="_blank">
          <img src={reactLogo} className="logo react" alt="React logo" />
        </a> */}
        <a href="https://react.dev" target="_blank">
          <ReactLogo2 className="logo react with plugin" />
        </a>

        <a href="https://rogerfederer.com" target="_blank">
          <img src={rfLogo} className="logo react" alt="RF logo" />
        </a>
      </div>
      <h1>Vite + React</h1>
      <h2>{greeting}</h2>
      <div className="card">
        <button onClick={() => setCount((count) => count + 1)}>
          count is {count}
        </button>
        <p>
          Edit <code>src/App.jsx</code> and save to test HMR
        </p>
      </div>
      <p className="read-the-docs">
        Click on the Vite and React logos to learn more
      </p>
      <Footer />
    </>
  )
}

export default App
