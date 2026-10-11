# Agentic Chatbot

A Streamlit chat UI backed by a LangGraph workflow and Google's Gemini model.

## Project files

```text
Agentic_chatbot/
├── .env.example                 # Environment variable template
├── .gitignore                   # Ignores credentials and local Python files
├── agentic_chatbot_backend.py   # LangGraph state, model, and compiled chatbot
├── app.py                       # Streamlit chat interface
├── app_thread.py                # Console streaming example
├── chatbot_workflow.ipynb       # Exploratory workflow notebook
├── requirements.txt             # Python dependencies
└── test.py                      # Existing standalone example
```

Open the `Agentic_chatbot` folder itself in VS Code using **File → Open Folder**. The files listed above should then appear in Explorer.

## Windows setup

Run these commands in a PowerShell terminal from the project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set `GEMINI_API_KEY` to a valid Google AI API key. Keep `.env` private; it is excluded from Git.

Start the web app:

```powershell
streamlit run app.py
```

To run the console streaming example instead:

```powershell
python app_thread.py
```

The notebook can be opened and run in VS Code after installing the Jupyter extension.
