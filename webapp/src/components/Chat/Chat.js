import React from 'react';
import './Chat.css';
import DoctorImage from './DoctorImage';

const Chat = () => {
  return (
    <div className="chat-container">
      <div className="doctor-image-container">
        <DoctorImage />
      </div>
      <div className="chat-interface">
        <div className="chat-messages">
          {/* Mensajes del chat */}
        </div>
        <div className="chat-input">
          <input type="text" placeholder="Escribe un mensaje..." />
          <button>Enviar</button>
        </div>
      </div>
    </div>
  );
};

export default Chat;
