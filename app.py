import os
import json
import datetime
import sqlite3
from pathlib import Path
from typing import Tuple, List

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_from_directory
import joblib
import numpy as np
import pandas as pd

# ML libs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.class_weight import compute_class_weight

# NLP utils
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import bcrypt
import urllib.parse

# Optional transformers 
USE_TRANSFORMERS = False  
if USE_TRANSFORMERS:
    import torch
    from transformers import (DistilBertTokenizerFast, DistilBertForSequenceClassification,
                              Trainer, TrainingArguments)

# -----------------------
# Config
# -----------------------
APP_ROOT = Path(__file__).parent.resolve()
MODEL_DIR = APP_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
DB_PATH = APP_ROOT / "users.db"
LOG_FILE = APP_ROOT / "prediction_logs.json"

# Saved model filenames
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
MODEL_PATH = MODEL_DIR / "clf_model.joblib"
META_PATH = MODEL_DIR / "model_meta.json"

# Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev_secret_change_me")

# -----------------------
# NLTK downloads (first run)
# -----------------------
nltk_data_path = str(APP_ROOT / "nltk_data")
os.environ.setdefault('NLTK_DATA', nltk_data_path)
nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
STOPWORDS = set(stopwords.words('english'))

# -----------------------
# Helper functions
# -----------------------
TRUSTED_DOMAINS = {
    # Big tech
    'google.com', 'gmail.com', 'youtube.com', 'microsoft.com', 'outlook.com', 'live.com',
    'apple.com', 'icloud.com', 'facebook.com', 'meta.com', 'instagram.com', 'whatsapp.com', 'linkedin.com',
    'amazon.com', 'amazon.in', 'aws.amazon.com', 'paypal.com', 'netflix.com', 'adobe.com',
    # Banks (examples; expand for your region)
    'chase.com', 'bankofamerica.com', 'wellsfargo.com', 'citi.com', 'hdfcbank.com', 'icicibank.com', 'sbi.co.in'
}
SUSPICIOUS_TLDS = {'xyz', 'top', 'click', 'link', 'app', 'work', 'gq', 'ml', 'tk', 'cf'}
PHISHING_HIGH_THRESHOLD = 0.7

def clean_text(text: str) -> str:
    """Basic cleaning: lower, remove urls, emails, punctuation, excessive whitespace, stopwords tokenized."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    # remove urls and emails
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    # remove non-alphanumeric (keep spaces)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)

def extract_urls(text: str) -> List[str]:
    if not text:
        return []
    url_regex = r"(https?://[\w\-\.:%#\?&/=]+|www\.[\w\-\.:%#\?&/=]+)"
    urls = re.findall(url_regex, text, flags=re.IGNORECASE)
    # Deduplicate while preserving order
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

def get_domain(url: str) -> str:
    try:
        if not url.lower().startswith(('http://', 'https://')):
            url = 'http://' + url
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname or ''
        return host.lower()
    except Exception:
        return ''

def base_registrable_domain(hostname: str) -> str:
    # Simple heuristic: join last two labels; for country TLDs this may be imperfect
    parts = hostname.split('.')
    if len(parts) >= 2:
        return '.'.join(parts[-2:])
    return hostname

def is_ip(hostname: str) -> bool:
    return bool(re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", hostname))

def analyze_url_legitimacy(text: str) -> dict:
    urls = extract_urls(text)
    findings = []
    risk_points = 0
    max_points = 1  # avoid zero-division; will increase as we add checks

    for u in urls:
        host = get_domain(u)
        base = base_registrable_domain(host)
        labels = host.split('.')
        tld = labels[-1] if labels else ''
        flags = []

        # Checks
        max_points += 4
        if is_ip(host):
            risk_points += 3
            flags.append('Uses raw IP address')
        if base not in TRUSTED_DOMAINS and not host.endswith(tuple(TRUSTED_DOMAINS)):
            risk_points += 1
            flags.append('Domain not in trusted list')
        if host.count('.') >= 3:
            risk_points += 1
            flags.append('Too many subdomains')
        if tld in SUSPICIOUS_TLDS:
            risk_points += 1
            flags.append(f'Suspicious TLD .{tld}')
        if re.search(r"\b(paypa1|g00gle|faceb00k|appl3)\b", host):
            risk_points += 2
            flags.append('Looks like a misspelled brand')

        findings.append({
            'url': u,
            'host': host,
            'base_domain': base,
            'flags': flags
        })

    # Rule-based score in [0,1]
    rule_score = min(1.0, risk_points / max(1, max_points))
    # Untrusted if at least one URL exists and none are trusted
    any_urls = len(findings) > 0
    trusted_flags = []
    for f in findings:
        base = f.get('base_domain', '')
        host = f.get('host', '')
        is_trusted = base in TRUSTED_DOMAINS or host.endswith(tuple(TRUSTED_DOMAINS))
        trusted_flags.append(is_trusted)
    url_all_untrusted = any_urls and all(not t for t in trusted_flags)
    return {'urls': findings, 'rule_score': float(rule_score), 'url_all_untrusted': url_all_untrusted}

def combine_scores(p_text: float, p_rules: float) -> float:
    # Weighted combination; adjust weights if needed
    return float(0.7 * p_text + 0.3 * p_rules)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL
    )''')
    conn.commit()
    conn.close()

def get_user_by_email(email: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE email = ?', (email,))
    user = c.fetchone()
    conn.close()
    return user

def create_user(name: str, email: str, password: str):
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('INSERT INTO users (name, email, password) VALUES (?, ?, ?)', (name, email, hashed))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def log_prediction(subject: str, content: str, prediction_label: str, confidence: float, model_name: str = "tfidf_lr"):
    entry = {
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'subject': subject,
        'content': content,
        'prediction': prediction_label,
        'confidence': round(float(confidence), 4),
        'model': model_name
    }
    logs = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r') as f:
                logs = json.load(f)
        except Exception:
            logs = []
    logs.append(entry)
    with open(LOG_FILE, 'w') as f:
        json.dump(logs, f, indent=2)

def save_model(vectorizer, model, meta: dict):
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(model, MODEL_PATH)
    with open(META_PATH, 'w') as f:
        json.dump(meta, f)

def load_model():
    if VECTORIZER_PATH.exists() and MODEL_PATH.exists():
        vect = joblib.load(VECTORIZER_PATH)
        model = joblib.load(MODEL_PATH)
        meta = {}
        if META_PATH.exists():
            with open(META_PATH, 'r') as f:
                meta = json.load(f)
        return vect, model, meta
    return None, None, None

# Initialize DB
init_db()

# If no model exists, create a lightweight baseline and save it
if not MODEL_PATH.exists():
    # Very small bootstrap dataset so app can run out of the box
    phishing_examples = [
        "Urgent: Your account has been suspended",
        "Verify your account immediately",
        "You've won a prize claim now",
        "Update your payment information",
        "Your account will be terminated"
    ]
    legit_examples = [
        "Team meeting at 3 PM tomorrow",
        "Project update: New features completed",
        "Thank you for your recent purchase",
        "Your monthly newsletter",
        "Invitation to company event"
    ]
    X = [clean_text(t) for t in (phishing_examples + legit_examples)]
    y = np.array([1] * len(phishing_examples) + [0] * len(legit_examples))
    vect = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    Xv = vect.fit_transform(X)
    clf = LogisticRegression(class_weight='balanced', max_iter=1000)
    clf.fit(Xv, y)
    meta = {"model": "tfidf_logreg_baseline", "trained_on": "bootstrapped_examples", "ngram_range": [1,2]}
    save_model(vect, clf, meta)
    print("Saved bootstrap model to", MODEL_DIR)

# Load model for inference
VECTORIZER, MODEL, META = load_model()

# -----------------------
# Routes
# -----------------------
@app.before_request
def require_login():
    allowed = ['login', 'signup', 'static', 'forgot_password', 'reset_password', 'api_predict', 'train_trigger']
    if request.endpoint not in allowed and not session.get('user_id'):
        return redirect(url_for('login'))

@app.route('/')
def home():
    return render_template('index.html')

# Form-based prediction (existing UI)
@app.route('/predict', methods=['POST'])
def predict():
    subject = request.form.get('subject', '')
    content = request.form.get('content', '')
    full_text = f"{subject} {content}"
    cleaned = clean_text(full_text)
    # ensure model available
    vect, model, meta = VECTORIZER, MODEL, META
    if vect is None or model is None:
        flash("Model not available. Please retrain using /train or upload a trained model.", "danger")
        return render_template('index.html', subject=subject, content=content)

    Xv = vect.transform([cleaned])
    proba = None
    url_analysis = analyze_url_legitimacy(full_text)
    p_rules = url_analysis.get('rule_score', 0.0)
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(Xv)[0]
            pred_idx = int(np.argmax(proba))
            p_text = float(proba[1])
            p_final = combine_scores(p_text, p_rules)
            confidence = p_final
        else:
            pred_idx = int(model.predict(Xv)[0])
            p_text = 1.0 if pred_idx == 1 else 0.0
            p_final = combine_scores(p_text, p_rules)
            confidence = p_final
    except Exception:
        pred_idx = int(model.predict(Xv)[0])
        p_text = 1.0 if pred_idx == 1 else 0.0
        p_final = combine_scores(p_text, p_rules)
        confidence = p_final

    # If URL analysis shows untrusted, force Phishing
    if url_analysis.get('url_all_untrusted'):
        label = "Phishing"
    else:
        label = "Phishing" if confidence >= PHISHING_HIGH_THRESHOLD else "Legitimate"
    log_prediction(subject, content, label, confidence, meta.get("model", "unknown"))
    
    # show top contributing features (for linear models)
    feature_explain = []
    if hasattr(model, "coef_"):
        try:
            coefs = model.coef_[0]
            feature_names = vect.get_feature_names_out()
            sorted_ids = np.argsort(coefs)[-10:]
            top_feats = [(feature_names[i], float(coefs[i])) for i in sorted_ids[::-1]]
            feature_explain = top_feats
        except Exception:
            feature_explain = []

    return render_template('index.html', prediction=label, confidence=f"{confidence*100:.2f}%", subject=subject, content=content, feature_explain=feature_explain, url_findings=url_analysis)

# JSON API for programmatic prediction
@app.route('/api/predict', methods=['POST'])
def api_predict():
    payload = request.get_json() or {}
    subject = payload.get('subject', '')
    content = payload.get('content', '')
    full_text = f"{subject} {content}"
    cleaned = clean_text(full_text)
    vect, model, meta = VECTORIZER, MODEL, META
    if vect is None or model is None:
        return jsonify({"error": "Model not available. Retrain first."}), 500
    Xv = vect.transform([cleaned])
    url_analysis = analyze_url_legitimacy(full_text)
    p_rules = url_analysis.get('rule_score', 0.0)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(Xv)[0]
        p_text = float(proba[1])
    else:
        pred_idx = int(model.predict(Xv)[0])
        p_text = 1.0 if pred_idx == 1 else 0.0
    p_final = combine_scores(p_text, p_rules)
    if url_analysis.get('url_all_untrusted'):
        label = "Phishing"
    else:
        label = "Phishing" if p_final >= PHISHING_HIGH_THRESHOLD else "Legitimate"
    log_prediction(subject, content, label, float(p_final), meta.get("model", "unknown"))
    return jsonify({
        "prediction": label,
        "scores": {
            "text_probability": p_text,
            "rule_score": p_rules,
            "combined": p_final
        },
        "url_findings": url_analysis.get('urls', []),
        "model": meta.get("model", "unknown")
    })

# View history logs
@app.route('/history')
def history():
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            logs = json.load(f)
    return render_template('history.html', logs=logs)

# Signup / Login / Logout
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name','').strip()
        email = request.form.get('email','').strip().lower()
        password = request.form.get('password','')
        confirm = request.form.get('confirm','')
        if not name or not email or not password or not confirm:
            flash('All fields required', 'danger')
            return render_template('signup.html')
        if password != confirm:
            flash('Passwords do not match', 'danger')
            return render_template('signup.html')
        if get_user_by_email(email):
            flash('Email already registered', 'danger')
            return render_template('signup.html')
        if create_user(name, email, password):
            flash('Registered successfully — log in', 'success')
            return redirect(url_for('login'))
        else:
            flash('Registration failed', 'danger')
    return render_template('signup.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email','').strip().lower()
        password = request.form.get('password','')
        remember = request.form.get('remember')
        user = get_user_by_email(email)
        if user and bcrypt.checkpw(password.encode('utf-8'), user[3]):
            session['user_id'] = user[0]
            session['user_name'] = user[1]
            session['user_email'] = user[2]
            session.permanent = bool(remember)
            return redirect(url_for('home'))
        else:
            flash('Invalid credentials', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out', 'success')
    return redirect(url_for('login'))

# Forgot/reset password (mock)
@app.route('/forgot-password', methods=['GET','POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email','').strip().lower()
        user = get_user_by_email(email)
        if user:
            session['reset_email'] = email
            session['reset_code'] = '123456'
            flash('Reset code sent (mock: 123456)', 'info')
            return redirect(url_for('reset_password'))
        else:
            flash('Email not found', 'danger')
    return render_template('forgot_password.html')

@app.route('/reset-password', methods=['GET','POST'])
def reset_password():
    if request.method == 'POST':
        code = request.form.get('code','')
        password = request.form.get('password','')
        confirm = request.form.get('confirm','')
        if code != session.get('reset_code'):
            flash('Invalid code', 'danger'); return render_template('reset_password.html')
        if password != confirm:
            flash('Passwords do not match', 'danger'); return render_template('reset_password.html')
        email = session.get('reset_email')
        if not email:
            flash('Session expired', 'danger'); return redirect(url_for('forgot_password'))
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('UPDATE users SET password = ? WHERE email = ?', (hashed, email))
        conn.commit()
        conn.close()
        session.pop('reset_email', None); session.pop('reset_code', None)
        flash('Password reset ok — log in', 'success')
        return redirect(url_for('login'))
    return render_template('reset_password.html')

# -----------------------
# Training endpoint (POST) - call this to retrain using a CSV
# Accepts JSON payload or form: {"csv_path": "/mnt/data/yourfile.csv", "model": "tfidf_lr" }
# -----------------------
def load_csv_for_training(path: str) -> Tuple[List[str], np.ndarray]:
    df = pd.read_csv(path, engine='python', on_bad_lines='warn')
    # attempt to find text + label columns
    cols = [c.lower() for c in df.columns]
    # heuristics:
    text_col = None
    label_col = None
    for c in df.columns:
        if 'text' in c or 'email' in c or 'body' in c or 'content' in c:
            text_col = c
            break
    for c in df.columns:
        if 'label' in c or 'type' in c or 'target' in c or 'class' in c:
            label_col = c
            break
    # fallback: first 2 columns
    if text_col is None:
        text_col = df.columns[0]
    if label_col is None:
        label_col = df.columns[1] if len(df.columns) > 1 else None
    if label_col is None:
        raise ValueError("Couldn't detect label column in CSV; provide a file with text + label columns")

    texts = df[text_col].fillna('').astype(str).apply(clean_text).tolist()
    labels = df[label_col].astype(str).str.lower().map(lambda x: 1 if 'phish' in x or '1' == x or x=='true' else 0).fillna(0).astype(int).values
    return texts, labels

def load_csv_for_training_raw(path: str) -> Tuple[List[str], np.ndarray]:
    df = pd.read_csv(path, engine='python', on_bad_lines='warn')
    text_col = None
    label_col = None
    for c in df.columns:
        if 'text' in c or 'email' in c or 'body' in c or 'content' in c or 'subject' in c:
            text_col = c
            break
    for c in df.columns:
        if 'label' in c or 'type' in c or 'target' in c or 'class' in c:
            label_col = c
            break
    if text_col is None:
        text_col = df.columns[0]
    if label_col is None:
        label_col = df.columns[1] if len(df.columns) > 1 else None
    if label_col is None:
        raise ValueError("Couldn't detect label column in CSV; provide a file with text + label columns")
    texts = df[text_col].fillna('').astype(str).tolist()
    labels = df[label_col].astype(str).str.lower().map(lambda x: 1 if 'phish' in x or '1' == x or x=='true' else 0).fillna(0).astype(int).values
    return texts, labels

class URLFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        rows = []
        for text in X:
            urls = re.findall(r'http[s]?://\S+|www\.\S+', text or '')
            url = urls[0] if urls else ""
            rows.append([
                len(url),
                url.count('.'),
                1 if '@' in url else 0,
                1 if url.startswith("https://") else 0,
                sum(ch in url for ch in ['-', '_', '?', '%', '#', '&', '='])
            ])
        return pd.DataFrame(rows)

def build_tfidf_url_lr_pipeline():
    tfidf = TfidfVectorizer(stop_words='english', max_features=20000, ngram_range=(1,2))
    url_feats = URLFeatures()
    return Pipeline([
        ("features", FeatureUnion([
            ("tfidf", tfidf),
            ("url", url_feats)
        ])),
        ("clf", LogisticRegression(max_iter=1000))
    ])

def generate_additional_ham_examples() -> List[str]:
    ham = [
        # Bank statements and OTPs
        "Your HDFC Bank statement for October is ready. Available balance: INR 53,421. OTP for adding beneficiary is 482913. Do not share with anyone.",
        "Chase Alert: You added a new device. If this was you, no action needed. For help visit chase.com/security.",
        # Legit login alerts
        "Google: Sign-in from a new Windows device. Was this you? Check your activity at myaccount.google.com.",
        "Microsoft Account security info changed successfully. If you didn't make this change, go to account.live.com.",
        # Social notifications
        "LinkedIn: Priya viewed your profile. See who else viewed your profile this week.",
        "Facebook: You have 3 new friend suggestions. Review them now.",
        # Subscriptions
        "Your Netflix subscription renews on Nov 28. Amount: $15.49. No action required.",
        "Adobe Creative Cloud: Payment received. Your plan is renewed for another month.",
        # Misc legitimate operational emails
        "Your GitHub personal access token will expire in 7 days. Visit settings to rotate the token.",
        "Outlook: Focused Inbox summary — 5 important emails from this week."
    ]
    return ham

@app.route('/train', methods=['GET','POST'])
def train_trigger():
    """
    GET: show small form
    POST: retrain model based on provided CSV path
    """
    global VECTORIZER, MODEL, META
    if request.method == 'POST':
        csv_path = request.form.get('csv_path') or request.json.get('csv_path')
        model_choice = request.form.get('model', 'tfidf_lr')
        augment_ham = bool(request.form.get('augment_ham'))
        if not csv_path:
            flash("Provide CSV path (absolute) to train", "danger")
            return render_template('train.html')

        try:
            if model_choice == 'tfidf_url_lr':
                texts, labels = load_csv_for_training_raw(csv_path)
            else:
                texts, labels = load_csv_for_training(csv_path)
        except Exception as e:
            flash(f"Failed to load CSV: {e}", "danger")
            return render_template('train.html')

        if augment_ham:
            extra_ham = generate_additional_ham_examples()
            texts.extend([clean_text(t) for t in extra_ham])
            labels = np.concatenate([labels, np.zeros(len(extra_ham), dtype=int)])

        # split
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.15, random_state=42, stratify=labels if len(np.unique(labels))>1 else None)
        # Build features/model
        if model_choice == 'tfidf_url_lr':
            # Use raw texts for this model
            model_pipeline = build_tfidf_url_lr_pipeline()
            model_pipeline.fit(X_train, y_train)
            preds = model_pipeline.predict(X_test)
            acc = accuracy_score(y_test, preds)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average='binary', zero_division=0)
            # Save entire pipeline
            joblib.dump(model_pipeline, MODEL_PATH)
            with open(META_PATH, 'w') as f:
                json.dump({"model": "tfidf_url_lr", "trained_on": os.path.basename(csv_path), "metrics": {"acc": acc, "precision": precision, "recall": recall, "f1": f1}}, f)
            # reload
            VECTORIZER, MODEL, META = None, joblib.load(MODEL_PATH), {"model": "tfidf_url_lr"}
            flash(f"Training complete — acc={acc:.4f}, f1={f1:.4f}. Model saved.", "success")
            return render_template('train.html', metrics={"acc": acc, "precision": precision, "recall": recall, "f1": f1}, csv_path=csv_path)
        else:
            # vectorize standard flow
            vect = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
            Xv_train = vect.fit_transform(X_train)
            Xv_test = vect.transform(X_test)

        # compute class weights
        if len(np.unique(y_train)) > 1:
            cw = 'balanced'
        else:
            cw = None

        if model_choice == 'multinb':
            clf = MultinomialNB()
            clf.fit(Xv_train, y_train)
            meta = {"model": "tfidf_multinb", "trained_on": os.path.basename(csv_path)}
        else:
            # default: logistic regression with simple grid search for C
            lr = LogisticRegression(class_weight=cw, max_iter=1000, solver='saga')
            params = {'C':[0.01, 0.1, 1, 10]}
            gs = GridSearchCV(lr, {'C':[0.01,0.1,1]}, cv=3, scoring='f1', n_jobs=1)
            gs.fit(Xv_train, y_train)
            clf = gs.best_estimator_
            meta = {"model": "tfidf_logreg", "trained_on": os.path.basename(csv_path), "best_params": gs.best_params_}

        # evaluate
        preds = clf.predict(Xv_test)
        acc = accuracy_score(y_test, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average='binary', zero_division=0)

        # save
        save_model(vect, clf, {"model": meta.get("model"), "trained_on": meta.get("trained_on"), "metrics": {"acc": acc, "precision": precision, "recall": recall, "f1": f1}})

        # reload into globals
        VECTORIZER, MODEL, META = load_model()

        flash(f"Training complete — acc={acc:.4f}, f1={f1:.4f}. Model saved.", "success")
        return render_template('train.html', metrics={"acc": acc, "precision": precision, "recall": recall, "f1": f1}, csv_path=csv_path)

    return render_template('train.html')

# Download model files (admin)
@app.route('/models/<path:filename>')
def serve_model_file(filename):
    return send_from_directory(MODEL_DIR, filename, as_attachment=True)

# -----------------------
# Run server
# -----------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)