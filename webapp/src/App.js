import React from 'react';
import './App.css';
import Header from './components/Header/Header';
import ImageDropzone from './components/ImageDropzone/ImageDropzone';
import Chat from './components/Chat/Chat';

function App() {
  return (
    <div className="App">
      <Header />
      <main className="main-content">
        <ImageDropzone />
        <Chat />
      </main>
    </div>
  );
}

export default App;
