# PhishGuard - Flask Phishing Detection App

A simple, production-friendly Flask app for phishing detection with:

- TF-IDF + Logistic Regression (default) and optional MultinomialNB
- Model persistence via joblib (vectorizer + model + meta)
- Retraining endpoint and UI from a CSV path
- Auth (signup/login/reset password)
- JSON prediction API and a form-based UI

## Requirements

- Python 3.12+
- Windows/macOS/Linux

Install Python deps:

```bash
pip install -r requirements.txt
```

Note: Transformers/Torch are not required. If you later want DistilBERT training, we can add it.

## Run the app

```bash
python app.py
```

Open `http://127.0.0.1:5000`.

On first run, a small bootstrap TF-IDF+LR model is created and saved under `models/`.

## Auth

- Signup: `/signup`
- Login: `/login`
- Logout: `/logout`
- Forgot/Reset Password: `/forgot-password`, `/reset-password` (mock code delivery)

## Train on your CSV

UI:
- Go to `/train`, enter absolute CSV path, choose model (LR or MultinomialNB), submit.

Programmatic (JSON):

```bash
curl -X POST http://127.0.0.1:5000/train \
  -H "Content-Type: application/json" \
  -d '{"csv_path": "C:/absolute/path/to/your.csv", "model": "tfidf_lr"}'
```

CSV expectations:
- One text column (heuristics try: text/email/body/content; fallback first col)
- One label column (heuristics try: label/type/target/class; fallback second col)
- Labels mapped to 1=phishing, 0=legit using simple rules

Saved files:
- `models/tfidf_vectorizer.joblib`
- `models/clf_model.joblib`
- `models/model_meta.json`

## Predict

Form UI:
- Use the home page to submit subject + content.

JSON API:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"subject":"Your account","content":"please verify your password"}'
```

Response:
```json
{"prediction":"Phishing","confidence":0.93,"model":"tfidf_logreg"}
```

## Logs

Predictions are appended to `prediction_logs.json`. View via `/history`.

## Structure

- `app.py` — Flask app with routes for auth, train, predict, history
- `templates/` — Tailwind UI templates
- `static/` — static assets
- `models/` — saved vectorizer/model/meta (auto-created)

## Notes

- For production: enable rate-limiting, CSRF, use `Flask-Login`, secure secret key, and run behind a WSGI server.
- If you want DistilBERT fine-tuning, say so and specify GPU availability.


