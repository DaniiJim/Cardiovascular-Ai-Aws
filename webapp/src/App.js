import React from 'react';
import './App.css';
import Header from './components/Header/Header';
import ImageDropzone from './components/ImageDropzone/ImageDropzone';
import Chat from './components/Chat/Chat';


function App() {
return (
<div className="App">
<Header />
<main className="main">
<section className="left-pane">
<div className="panel card">
<h2 className="panel-title">Upload image</h2>
<p className="panel-sub">Drag and drop or click to select</p>
<ImageDropzone />
</div>
</section>
<section className="right-pane">
<div className="panel card">
<div className="panel-title-row">
<h2 className="panel-title">Chat asistente</h2>
<span className="badge">Beta</span>
</div>
<Chat />
</div>
</section>
</main>
<footer className="footer">
<div className="footer-inner">© {new Date().getFullYear()} Cardiovascular AI · Inspirado en Amazon</div>
</footer>
</div>
);
}


export default App;