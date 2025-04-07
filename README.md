# Python Environment Setup

## Execute the following commands in the project directory:

### 1. Copy the configuration file:
`cp ./cse476.yaml ~/cse476.yaml`
### 2. Navigate to the home directory:
`cd ~`
### 3. Load the Mamba module:
`module load mamba/latest`
### 4. Create the Python environment:
`mamba env create -n cse476 --file ~/cse476.yaml`
### 5. Activate the environment:
`source activate cse476`

# Hugging Face Transformers Setup
### 1. Create an account on Hugging Face.
### 2. Request access to `llama-3.2-3b` (approval may take about 10 minutes).
![Access](./img/access.png)
    You can check the request status under Settings -> Gated Repo.
![Status](./img/status.png)
### 3. Obtain a Hugging Face Access Token (Settings -> Get Access Token).
### 4. Copy the template configuration file:
`cp ./Backend/config_temp.py ./Backend/config.py`
### 5. Paste your Access Token into `config.py`.

# Dependency Library Installation
### 1. After activating the Python environment, you should see `(cse476)` in your terminal.
![Environment](./img/env.png)
### 2. Install the required libraries:
`pip install -r requirements.txt`
### 3. Once the installation is complete, the code is ready to run.

# Test LLM API
### 1. Open sol.asu.edu.
### 2. Create a new desktop session.
![desktop](./img/desktop.png)
### 3. Open the desktop session.
![open-desktop](./img/open_desktop.png)
### 4. Open two terminals in the desktop session and activate the `cse476` Python environment in both terminals.
### 5. In the **Backend** folder, run the following command in the first terminal (this starts the LLM API server):
`uvicorn main:app --host 0.0.0.0 --port 8000`
### 6. In the **frontend** folder, run the following command in the second terminal (this starts the client):
`python server.py`
### 7. Open Firefox and navigate to:
http://127.0.0.1:5000 or http://localhost:5000
### 8. You will see a chat interface where you can test the application.