import React, { useState } from 'react';

function App() {
  const [question, setQuestion] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResponse(''); // clear previous response

    try {
      const res = await fetch('http://localhost:8000/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ prompt: question })
      });

      if(res.ok)
      {
        const data = await res.json();
        setResponse(data.response);
      }
      else
      {
        setResponse('Failed to fetch response from the server.');
      }
    } catch (error) {
      console.error('Error:', error);
      setResponse('An error occurred. Please try again.');
    }
    setLoading(false);
  };

  return (
    <div style={{ padding: '20px' }}>
      <h1>Ask a Question</h1>
      <form onSubmit={handleSubmit}>
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          rows={4}
          placeholder="Enter your question here..."
          style={{ width: '100%', padding: '10px', fontSize: '16px' }}
        />
        <br />
        <button type="submit" 
                disabled={loading} 
                style={{ marginTop: '10px', padding: '10px 20px', fontSize: '16px' }}
        >
          {loading ? 'Submitting...' : 'Submit'}
        </button>
      </form>
      {response && (
        <div style={{ marginTop: '20px', backgroundColor: '#f9f9f9', padding: '15px', border: '1px solid #ccc' }}>
          <h2>Response:</h2>
          <p>{response}</p>
        </div>
      )}
    </div>
  );
}

export default App;

