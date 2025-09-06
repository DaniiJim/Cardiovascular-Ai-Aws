import React from 'react';
import './Header.css';

const Header = () => {
  return (
    <header className="aws-amz-header">
      <div className="topbar">
        <div className="container">
          <div className="left">
            <div className="logo" aria-label="Logo">🩺</div>
            <div className="brand">
              <div className="brand-title">Cardiovascular AI</div>
              <div className="brand-sub">Smartwatch ECG explanation</div>
            </div>
          </div>
        </div>
      </div>

      <div className="subbar">
        <div className="container sub">
          <nav className="nav">
            <a href="#" className="nav-link">Subir ECG</a>
            <a href="#" className="nav-link">Explicación LLM</a>
            <a href="#" className="nav-link">Soporte</a>
          </nav>
        </div>
      </div>
    </header>
  );
};

export default Header;
