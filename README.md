# Machine Learning Based Intrusion Detection System (IDS)

This project implements the **Machine Learning based Intrusion Detection System** described in the provided report:
- Data preprocessing (cleaning, encoding, scaling)
- XGBoost model training and evaluation
- Intrusion detection on uploaded datasets
- Visualization dashboard with email + password login
- PDF report generation + optional email delivery (SMTP)
- Optional live monitoring (packet sniffing) with a safe fallback simulation

## Quickstart

### 1) Setup

```bash
cd /home/dell/project-endsem
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) One-command demo setup (NSL-KDD)

This will download NSL-KDD CSVs, create `data/train.csv` + `data/test.csv` with a `label` column, and train the model.

```bash
python scripts/setup_demo.py
```

### 3) Train a model (using your dataset CSV)

Put a CSV dataset at `data/train.csv` (or provide a path). The CSV must include a label column:
- default label column name: `label`
- label values: `normal` / `attack` (case-insensitive), or `0/1`

```bash
python -m ids.train --data data/train.csv --label-col label --out models/ids_model.joblib
```

### 4) Run the dashboard

```bash
streamlit run app/streamlit_app.py
```

### Login / Sign up

- **Sign up**: create an account with email + password (stored locally in `data/app_users.json` with hashed passwords).
- **Login**: use the same email/password.

Optional fixed “single user” login via env / Streamlit secrets (useful on Streamlit Cloud if you don’t want local user storage):

- `IDS_USER`: email
- `IDS_PASS`: password

```bash
IDS_USER=you@example.com IDS_PASS=yourpass streamlit run app/streamlit_app.py
```

### 5) PDF report + email (SMTP)

The dashboard can generate a PDF and optionally email it.

Set these environment variables (or Streamlit Cloud **Secrets**):

- `IDS_SMTP_HOST` (e.g. `smtp.gmail.com`, or Brevo `smtp-relay.brevo.com`)
- `IDS_SMTP_PORT` (usually `587` for STARTTLS, or `465` for SSL)
- `IDS_SMTP_USER` (your SMTP username)
- `IDS_SMTP_PASS` (app password / SMTP password)
- `IDS_SMTP_FROM` (the From address; often same as `IDS_SMTP_USER`)
- `IDS_SMTP_STARTTLS` (default `1`; set `0` if your provider does not use STARTTLS)
- `IDS_SMTP_SSL` (set `1` to force `SMTP_SSL` on non-465 ports; usually not needed)

Brevo (example):

```toml
IDS_SMTP_HOST="smtp-relay.brevo.com"
IDS_SMTP_PORT="587"
IDS_SMTP_USER="xxxx@yourdomain.com"        # Brevo SMTP login (shown in Brevo SMTP settings)
IDS_SMTP_PASS="YOUR_BREVO_SMTP_KEY"        # Brevo SMTP key (not your web login password)
IDS_SMTP_FROM="xxxx@yourdomain.com"        # Must be a verified sender in Brevo
IDS_SMTP_STARTTLS="1"
```

If you use port `465`, you typically do not need `STARTTLS` (the app will use implicit TLS automatically when port is 465).

## Data notes (NSL-KDD / CICIDS2017)

The report references NSL-KDD and CICIDS2017. Those datasets are large and not bundled here.
You can export them to CSV and point `ids.train` to the CSV.

## Project structure

- `ids/`: core package (preprocess, model, predict, live monitoring helpers)
- `app/`: Streamlit dashboard (auth, upload, charts, live view)
- `models/`: saved model artifacts (created after training)
- `data/`: your datasets (not committed by default)

## Commands

- Train: `python -m ids.train --data data/train.csv --label-col label`
- Predict CSV: `python -m ids.predict --model models/ids_model.joblib --data data/test.csv`

