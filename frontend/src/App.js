import React, { useState } from 'react';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    const newMessages = [...messages, { type: 'user', text: question }];
    setMessages(newMessages);
    setQuestion('');
    setLoading(true);

    try {
      const res = await fetch('http://localhost:8000/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ prompt: question })
      });

      if (res.ok) {
        const data = await res.json();
        setMessages([...newMessages, { type: 'bot', text: data.response }]);
      } else {
        setMessages([...newMessages, { type: 'bot', text: 'Failed to fetch response from the server.' }]);
      }
    } catch (error) {
      console.error('Error:', error);
      setMessages([...newMessages, { type: 'bot', text: 'An error occurred. Please try again.' }]);
    }
    setLoading(false);
  };

  return (
    <div className="chat-container">
      <div className="chat-window">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.type}`}>{msg.text}</div>
        ))}
      </div>
      <form className="chat-input" onSubmit={handleSubmit}>
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          rows={2}
          placeholder="Type your message..."
          disabled={loading}
        />
        <button type="submit" disabled={loading}>{loading ? 'Sending...' : 'Send'}</button>
      </form>
    </div>
  );
}

export default App;

