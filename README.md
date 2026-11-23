# AgroIntel AI

Smart soil analysis, crop recommendation, yield prediction, sustainability,
and optional Gemini-powered agricultural assistance in a Streamlit dashboard.

## Run in VS Code

The project is configured to use `.venv` automatically.

1. Open this folder in VS Code.
2. Choose **Terminal > Run Task > Run AgroIntel AI**.
3. Open the local URL shown in the terminal (normally `http://localhost:8501`).

You can also run it from the VS Code terminal:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Optional Gemini features

Crop prediction and the other local features work without an API key. For AI
chat and image analysis, copy `.streamlit/secrets.toml.example` to
`.streamlit/secrets.toml`, add a valid Google Gemini API key, and restart the
app. Never commit `secrets.toml`.

## Rebuild the environment

If `.venv` is removed, create a Python virtual environment and install:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r assets\requirements.txt
```
