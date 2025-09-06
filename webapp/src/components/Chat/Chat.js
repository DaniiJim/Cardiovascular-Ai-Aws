import React, { useState } from 'react';
import './Chat.css';
import DoctorImage from './DoctorImage';


const Chat = () => {
const [messages, setMessages] = useState([
{ role: 'assistant', content: 'Hi, I am your virtual doctor. How can I help you?' }
]);
const [text, setText] = useState('');


const send = () => {
const t = text.trim();
if (!t) return;
setMessages(prev => [...prev, { role: 'user', content: t }]);
setText('');
// Aquí podrás invocar tu backend/LLM y hacer setMessages con la respuesta
};


return (
<div className="chat">
<aside className="doc">
<DoctorImage />
</aside>
<section className="thread">
<div className="msgs">
{messages.map((m, i) => (
<div key={i} className={`msg ${m.role}`}>
<div className="bubble">{m.content}</div>
</div>
))}
</div>
<div className="composer">
<input
value={text}
onChange={e => setText(e.target.value)}
onKeyDown={e => e.key === 'Enter' && send()}
placeholder="Type your message…"
/>
<button onClick={send}>Send</button>
</div>
</section>
</div>
);
};


export default Chat;