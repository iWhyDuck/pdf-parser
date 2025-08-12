# PDF Parser Pro

AI-powered PDF analysis with field extraction, history tracking and rich Streamlit UI.

---

## Features
- Multiple text-extraction back-ends (`PyPDF2`, `pdfplumber`, `PyMuPDF`, native OCR)
- Vision + text pipeline powered by OpenAI (`gpt-4-vision-preview`, `gpt-4-turbo`)
- Structured extraction jobs stored in SQLite (via SQLAlchemy)
- Streamlit interface for batch uploads, quick scans, detailed extraction & history
- Modern packaging with **Poetry**

---

## Requirements
- **Python ≥ 3.10**
- A valid `OPENAI_API_KEY` in your environment  

---

## Getting started

### 1. Clone and enter the project
```bash
git clone https://github.com/your-user/pdf-parser.git
cd pdf-parser
```

### 2. Install Poetry ≥ 2.0 (if needed)
```bash
pipx install poetry  # or `pip install --user poetry`
```

### 3. Install dependencies
```bash
poetry install
```

### 4. Activate the virtual environment  

Since Poetry `>= 2.0.0` the legacy `poetry shell` command is **not** installed by default.

Choose **one** of the following options:

1. **Recommended (activate)**  
   ```bash
   # create / select a Python 3.10 env (only once)
   poetry env use python3.10

   # show available environments with their full paths
   poetry env list --full-path

   # activate the one marked "(activated)" or copy the path you prefer
   poetry env activate <path-shown-above>
   ```

2. **Optional (shell plugin)**  
   If you miss the old `poetry shell` convenience command:
   ```bash
   poetry self add poetry-plugin-shell   # installs the plugin
   poetry shell                          # now works again
   ```

### 5. Run the Streamlit app
```bash
# inside the activated environment
streamlit run temp.py           # or adapt to your entry-point
```

---

## Common tasks
| Task | Command |
|------|---------|
| Run tests | `poetry run pytest` |
| Format code | `poetry run black .` |
| Static typing | `poetry run mypy src` |
| Update deps | `poetry update` |

---

## Environment variables
Set your OpenAI key **before** launching:
```bash
export OPENAI_API_KEY="sk-…"
```
(You can also use a `.env` file with `python-dotenv` support.)

---

## License
MIT