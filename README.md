# python environment setting

## Following bash commands should be executed in the project directory

### 1. `cp ./cse476.yaml ~/cse476.yaml`
### 2. `cd ~`
### 3. `module load mamba/latest`
### 4. `mamba env create -n cse476 --file ~/cse476.yaml`
### 5. `source activate cse476`

# Hugging Face transformers setting
### 1. Create an Account on Hugging Face
### 2. Grant Access to llama-3.2-3b (takes about 10 minutes to get access)
![Access](./img/access.png)
    You can check the request status (Settings -> Gated Repo)
![Status](./img/status.png)
### 3. Get a Hugging Face Access Token (Settings -> Get Access Token)
### 4. Copy `config_temp.py` to `config.py` (cp ./Backend/config_temp.py ./Backend/config.py)
### 5. Copy and paste your Access Token into `config.py`

# Dependency library installation
### 1. Once you successfully open the Python environment, you will see `(cse476)` in your terminal.
![Access](./img/env.png)
### 2. Now you can download the dependency libraries to run the code.
### 3. Run `pip install -r requirements.txt`
### 4. Once the download is completed, you are ready to run the code.

# Test LLM API
### 1. Open two terminals.
### 2. Run `python main.py` in the first terminal (this is the LLM API server).
### 3. Run `python test_llm.py` in the second terminal (this is the client).