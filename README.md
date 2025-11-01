# BIA601 Feature Selection App (Flask)

This app lets you upload a CSV dataset and compare Genetic Algorithm (GA) feature selection with traditional methods (e.g., VarianceThreshold) using the same train/test split. The UI shows selected features, accuracy, evolution history, and execution time.

## Requirements

- Python 3.9+ (recommended)
- pip (comes with Python)

All Python dependencies are listed in `requirements.txt`.

## Quick Start (Linux/macOS)

1. Create a virtual environment in the project root:

```bash
python3 -m venv .venv
```

2. Activate it:

```bash
source .venv/bin/activate
```

3. Upgrade pip and install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

4. Run the app:
   
```bash
gunicorn app:app --workers 4
```
or 

```bash
python app.py
```

5. Open your browser at:

```
http://127.0.0.1:5000/
```

6. Deactivate the venv when done:

```bash
deactivate
```

## Quick Start (Windows)

1. Create a virtual environment in the project root:

```bat
py -3 -m venv .venv
```

2. Activate it (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

- If you see an execution policy error, run PowerShell as a regular user and execute:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

3. Upgrade pip and install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

4. Run the app:

```powershell
gunicorn app:app --workers 4
```

or 

```powershell
python app.py
```

5. Open your browser at:

```
http://127.0.0.1:5000/
```

6. Deactivate the venv when done:

```powershell
deactivate
```

## Run Tests

With the virtual environment active:

```bash
pytest -q
```
