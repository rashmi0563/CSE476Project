from flask import Flask, send_from_directory, request, jsonify
import requests

app = Flask(__name__, static_folder='build', static_url_path='')

BACKEND_URL = "http://localhost:8000/generate"

@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/generate', methods=['POST'])
def generate():
    data = request.get_json()
    response = requests.post(BACKEND_URL, json=data)
    return jsonify(response.json())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)