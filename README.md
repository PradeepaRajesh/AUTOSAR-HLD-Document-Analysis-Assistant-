# AUTOSAR HLD Document Analysis Assistant

## Setup
1. Install Python 3.10+ and Ollama (ollama.com), then: `ollama pull llama3.2:3b`
2. `python -m venv venv` and activate it (Windows: `venv\Scripts\activate`; Mac/Linux: `source venv/bin/activate`)
3. `pip install -r requirements.txt`
4. Put a public AUTOSAR PDF in `Input_Data/`

## Run
```
python Code/ingest.py Input_Data/yourfile.pdf     # or upload inside the app
streamlit run Code/app.py
python Code/evaluate.py                           # after filling Evaluation_Results/eval_questions.csv
```

## Slow laptop? Use a free hosted LLM instead of Ollama
```
# Mac/Linux
export LLM_BACKEND=api API_KEY=your_key API_BASE=https://api.groq.com/openai/v1 API_MODEL=llama-3.1-8b-instant
# Windows PowerShell
$env:LLM_BACKEND="api"; $env:API_KEY="your_key"
```
Declare it in the AI-tool table (the document is public, so this is acceptable).

## Tuning
Edit `Code/config.py`. If out-of-scope questions are answered, lower THRESHOLD; if valid questions are refused, raise it (check `top_distance` in eval_results.csv).
