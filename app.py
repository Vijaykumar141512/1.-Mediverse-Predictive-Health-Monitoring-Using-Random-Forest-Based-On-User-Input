from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file
from fpdf import FPDF
import io
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import os
import json
import subprocess
import sys

app = Flask(__name__)
app.config['SECRET_KEY'] = 'health_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///health.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    histories = db.relationship('PredictionHistory', backref='user', lazy=True)

class Admin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    body_temperature = db.Column(db.Float, nullable=False)
    steps = db.Column(db.Integer, nullable=False)
    health_index = db.Column(db.Integer, nullable=False)
    flu_information = db.Column(db.Integer, nullable=False)
    heart_rate = db.Column(db.Integer, nullable=False)
    oxygen_level = db.Column(db.Integer, nullable=False)
    result = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

def create_db():
    with app.app_context():
        db.create_all()
        # Initialize an admin
        if not Admin.query.filter_by(username='admin').first():
            admin = Admin(username='admin', password=generate_password_hash('admin'))
            db.session.add(admin)
            db.session.commit()

# Ensure database exists
if not os.path.exists('instance/health.db'):
    create_db()

# Helper function to get XAI suggestions
def get_xai_suggestions(result):
    if result.lower() == 'risk':
        return [
            "Your body exhibits critical signals matching our high-risk profiles.",
            "Elevated temperature and abnormal heart rate/oxygen patterns heavily influenced this result.",
            "Action: Immediately consult a doctor. Focus on resting and seeking emergency medical help if breathing is difficult."
        ]
    elif result.lower() == 'medium':
        return [
            "Your health indicators are borderline. Your physical activity (steps) or vitals are suboptimal.",
            "Our model flagged your inputs as similar to individuals developing minor illnesses.",
            "Action: Increase hydration, take rest, and monitor your vitals closely for the next 48 hours."
        ]
    else:
        return [
            "All vital signs are within normal, healthy ranges.",
            "Your physical activity and heart rate indicate a strong, stable physiological state.",
            "Action: Continue your current healthy lifestyle and maintain daily steps."
        ]


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email address already exists')
            return redirect(url_for('signup'))
            
        new_user = User(name=name, email=email, password=generate_password_hash(password))
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
        
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password, password):
            flash('Please check your login details and try again.')
            return redirect(url_for('login'))
            
        session['user_id'] = user.id
        session['user_name'] = user.name
        return redirect(url_for('user_dashboard'))
        
    return render_template('login.html')

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        admin = Admin.query.filter_by(username=username).first()
        if not admin or not check_password_hash(admin.password, password):
            flash('Invalid admin credentials.')
            return redirect(url_for('admin_login'))
            
        session['admin_id'] = admin.id
        return redirect(url_for('admin_dashboard'))
        
    return render_template('admin_login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/user_dashboard', methods=['GET', 'POST'])
def user_dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    user = User.query.get(session['user_id'])
    prediction_result = None
    suggestions = None
    
    if request.method == 'POST':
        try:
            body_temperature = float(request.form.get('body_temperature'))
            steps = int(request.form.get('steps'))
            health_index = int(request.form.get('health_index'))
            flu_information = int(request.form.get('flu_information'))
            heart_rate = int(request.form.get('heart_rate'))
            oxygen_level = int(request.form.get('oxygen_level'))
            
            # Load Random Forest model
            model_path = os.path.join(os.path.dirname(__file__), 'models/rf_model.pkl')
            if not os.path.exists(model_path):
                flash("Models not trained yet! Please ask Admin to train the models.")
                return redirect(url_for('user_dashboard'))
                
            with open(model_path, 'rb') as f:
                rf_model = pickle.load(f)
                
            # Fix for scikit-learn 1.4+ compatibility with older models
            if hasattr(rf_model, 'estimators_'):
                for tree in rf_model.estimators_:
                    if not hasattr(tree, 'monotonic_cst'):
                        tree.monotonic_cst = None
                
            # Predict
            features = np.array([[body_temperature, steps, health_index, flu_information, heart_rate, oxygen_level]])
            prediction_result = rf_model.predict(features)[0]
            
            # XAI Suggestions
            suggestions = get_xai_suggestions(prediction_result)
            
            # Save history
            new_history = PredictionHistory(
                user_id=user.id,
                body_temperature=body_temperature,
                steps=steps,
                health_index=health_index,
                flu_information=flu_information,
                heart_rate=heart_rate,
                oxygen_level=oxygen_level,
                result=prediction_result
            )
            db.session.add(new_history)
            db.session.commit()
        except Exception as e:
            flash(f"Error making prediction: {str(e)}")
            
    return render_template('user_dashboard.html', user=user, prediction_result=prediction_result, suggestions=suggestions)

@app.route('/admin_dashboard')
def admin_dashboard():
    if 'admin_id' not in session:
        return redirect(url_for('admin_login'))
        
    users = User.query.all()
    all_history = PredictionHistory.query.join(User).order_by(PredictionHistory.timestamp.desc()).all()
    
    metrics = None
    metrics_path = os.path.join(os.path.dirname(__file__), 'models/metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.loads(f.read())
            
    dataset_details = ""
    ds_path = os.path.join(os.path.dirname(__file__), 'dataset/health_dataset.csv')
    if os.path.exists(ds_path):
        df = pd.read_csv(ds_path)
        dataset_details = f"Loaded {len(df)} records. Classes: {', '.join(df['target'].unique())}."
            
    return render_template('admin_dashboard.html', users=users, metrics=metrics, dataset_details=dataset_details, history=all_history)

@app.route('/train_model', methods=['POST'])
def train_model():
    if 'admin_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    model_type = request.form.get('model_type')
    if model_type not in ['SVM', 'Linear', 'KNN', 'RandomForest']:
        return jsonify({'error': 'Invalid model type'}), 400
        
    script_path = os.path.join(os.path.dirname(__file__), 'model_training.py')
    try:
        subprocess.run([sys.executable, script_path], check=True)
        
        # Read the metrics to return specifically for this model
        metrics_path = os.path.join(os.path.dirname(__file__), 'models/metrics.json')
        model_metrics = {}
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                all_metrics = json.loads(f.read())
                model_metrics = all_metrics.get(model_type, {})

        return jsonify({
            'message': f'{model_type} trained successfully!',
            'metrics': model_metrics,
            'model_name': model_type
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train_all_models', methods=['POST'])
def train_all_models():
    if 'admin_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
        
    script_path = os.path.join(os.path.dirname(__file__), 'model_training.py')
    try:
        # Run the training script which trains all models
        subprocess.run([sys.executable, script_path], check=True)
        
        # Read the metrics file
        metrics_path = os.path.join(os.path.dirname(__file__), 'models/metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                all_metrics = json.loads(f.read())
            return jsonify({
                'message': 'All models (SVM, Linear, KNN, Random Forest) trained successfully!',
                'metrics': all_metrics
            })
        else:
            return jsonify({'error': 'Metrics file not found after training'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    if 'admin_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
        
    if 'dataset_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['dataset_file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        ds_dir = os.path.join(os.path.dirname(__file__), 'dataset')
        if not os.path.exists(ds_dir):
            os.makedirs(ds_dir)
        
        ds_path = os.path.join(ds_dir, 'health_dataset.csv')
        file.save(ds_path)
        
        # Verify the file is a CSV and has data
        df = pd.read_csv(ds_path)
        return jsonify({'message': f'Dataset "{file.filename}" uploaded successfully! Loaded {len(df)} records.'})

@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
def delete_user(user_id):
    if 'admin_id' not in session:
        return redirect(url_for('admin_login'))
    user = User.query.get_or_404(user_id)
    # Delete associated history first
    PredictionHistory.query.filter_by(user_id=user.id).delete()
    db.session.delete(user)
    db.session.commit()
    flash(f"User {user.name} removed successfully.")
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/add_user', methods=['POST'])
def add_user():
    if 'admin_id' not in session:
        return redirect(url_for('admin_login'))
    name = request.form.get('name')
    email = request.form.get('email')
    password = request.form.get('password')
    
    if User.query.filter_by(email=email).first():
        flash('Email already exists')
    else:
        new_user = User(name=name, email=email, password=generate_password_hash(password))
        db.session.add(new_user)
        db.session.commit()
        flash(f"User {name} added successfully.")
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/download/csv')
def download_all_csv():
    if 'admin_id' not in session:
        return redirect(url_for('admin_login'))
    
    history = PredictionHistory.query.join(User).all()
    data = []
    for h in history:
        data.append({
            'User': h.user.name,
            'Email': h.user.email,
            'Temperature': h.body_temperature,
            'Steps': h.steps,
            'Health Index': h.health_index,
            'Flu': h.flu_information,
            'Heart Rate': h.heart_rate,
            'Oxygen': h.oxygen_level,
            'Result': h.result,
            'Timestamp': h.timestamp
        })
    df = pd.DataFrame(data)
    output = io.BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return send_file(output, mimetype='text/csv', as_attachment=True, download_name='user_risk_data.csv')

@app.route('/admin/download/pdf')
def download_all_pdf():
    if 'admin_id' not in session:
        return redirect(url_for('admin_login'))
    
    history = PredictionHistory.query.join(User).all()
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(190, 15, "Global Health Prediction Master Report", ln=True, align='C')
    pdf.ln(5)
    
    pdf.set_font("Arial", "B", 10)
    pdf.set_text_color(0, 0, 0)
    pdf.set_fill_color(220, 230, 241)
    
    # Headers - Comprehensive
    headers = ['User', 'Temp', 'Steps', 'Index', 'Flu', 'HR', 'O2', 'Result', 'Date']
    col_widths = [30, 15, 20, 15, 12, 15, 15, 25, 43]
    
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 10, header, border=1, fill=True, align='C')
    pdf.ln()
    
    pdf.set_font("Arial", "", 8)
    for h in history:
        pdf.cell(col_widths[0], 8, h.user.name[:18], border=1)
        pdf.cell(col_widths[1], 8, str(h.body_temperature), border=1, align='C')
        pdf.cell(col_widths[2], 8, str(h.steps), border=1, align='C')
        pdf.cell(col_widths[3], 8, str(h.health_index), border=1, align='C')
        pdf.cell(col_widths[4], 8, 'Y' if h.flu_information == 1 else 'N', border=1, align='C')
        pdf.cell(col_widths[5], 8, str(h.heart_rate), border=1, align='C')
        pdf.cell(col_widths[6], 8, str(h.oxygen_level), border=1, align='C')
        
        # Color code result
        if h.result.lower() == 'risk':
            pdf.set_text_color(200, 0, 0)
        elif h.result.lower() == 'medium':
            pdf.set_text_color(200, 100, 0)
        else:
            pdf.set_text_color(0, 120, 0)
            
        pdf.cell(col_widths[7], 8, h.result.upper(), border=1, align='C')
        pdf.set_text_color(0, 0, 0)
        pdf.cell(col_widths[8], 8, h.timestamp.strftime('%Y-%m-%d %H:%M'), border=1, align='C')
        pdf.ln()
        
    output = io.BytesIO()
    pdf_content = pdf.output(dest='S')
    if isinstance(pdf_content, (bytes, bytearray)):
        output.write(pdf_content)
    else:
        output.write(pdf_content.encode('latin-1'))
    output.seek(0)
    return send_file(output, mimetype='application/pdf', as_attachment=True, download_name='all_users_health_report.pdf')

@app.route('/user/download_prediction/<int:history_id>')
def download_user_pdf(history_id):
    if 'user_id' not in session and 'admin_id' not in session:
        return redirect(url_for('login'))
        
    h = PredictionHistory.query.get_or_404(history_id)
    if 'user_id' in session and h.user_id != session['user_id']:
        return "Unauthorized", 401
        
    pdf = FPDF()
    pdf.add_page()
    
    # Border
    pdf.rect(5, 5, 200, 287)
    
    pdf.set_font("Arial", "B", 22)
    pdf.set_text_color(0, 102, 204)
    pdf.cell(190, 20, "Health Prediction Detailed Report", ln=True, align='C')
    
    pdf.set_draw_color(0, 102, 204)
    pdf.line(20, 35, 190, 35)
    pdf.ln(10)
    
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(95, 10, f"Patient Name: {h.user.name}", ln=0)
    pdf.cell(95, 10, f"Report Date: {h.timestamp.strftime('%Y-%m-%d %H:%M')}", ln=1, align='R')
    pdf.ln(5)
    
    # Vital Signs Table-like structure
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(190, 10, "  Reported Health Metrics", ln=True, fill=True)
    pdf.ln(2)
    
    pdf.set_font("Arial", "", 12)
    metrics = [
        ("Body Temperature", f"{h.body_temperature} F"),
        ("Daily Steps Walked", f"{h.steps}"),
        ("Overall Health Index", f"{h.health_index} / 100"),
        ("Flu Symptoms Present", "Yes" if h.flu_information == 1 else "No"),
        ("Heart Rate", f"{h.heart_rate} bpm"),
        ("Oxygen Saturation (SpO2)", f"{h.oxygen_level}%")
    ]
    
    for label, value in metrics:
        pdf.set_font("Arial", "B", 11)
        pdf.cell(80, 10, f"  {label}:", ln=0)
        pdf.set_font("Arial", "", 11)
        pdf.cell(110, 10, value, ln=1)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    
    pdf.ln(10)
    
    # Result Box
    pdf.set_font("Arial", "B", 16)
    if h.result.lower() == 'risk':
        pdf.set_fill_color(255, 230, 230)
        pdf.set_text_color(200, 0, 0)
        border_col = (200, 0, 0)
    elif h.result.lower() == 'medium':
        pdf.set_fill_color(255, 245, 230)
        pdf.set_text_color(200, 100, 0)
        border_col = (200, 100, 0)
    else:
        pdf.set_fill_color(230, 255, 230)
        pdf.set_text_color(0, 120, 0)
        border_col = (0, 120, 0)
        
    pdf.set_draw_color(*border_col)
    pdf.cell(190, 15, f"AI PREDICTION: {h.result.upper()}", border=1, ln=True, align='C', fill=True)
    
    pdf.ln(10)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(190, 10, "Explainable AI (XAI) Health Guidance:", ln=True)
    
    pdf.set_font("Arial", "", 11)
    suggestions = get_xai_suggestions(h.result)
    for s in suggestions:
        pdf.multi_cell(190, 8, f"- {s}", border=0)
        pdf.ln(1)
        
    # Footer
    pdf.set_y(-30)
    pdf.set_font("Arial", "I", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(190, 10, "This is an AI-generated health prediction report. Please consult a medical professional for clinical diagnosis.", ln=True, align='C')
    
    output = io.BytesIO()
    pdf_content = pdf.output(dest='S')
    if isinstance(pdf_content, (bytes, bytearray)):
        output.write(pdf_content)
    else:
        output.write(pdf_content.encode('latin-1'))
    output.seek(0)
    return send_file(output, mimetype='application/pdf', as_attachment=True, download_name=f'Health_Report_{h.id}.pdf')

@app.route('/admin/user_report/<int:user_id>')
def admin_download_user_report(user_id):
    if 'admin_id' not in session:
        return redirect(url_for('admin_login'))
    
    user = User.query.get_or_404(user_id)
    history = PredictionHistory.query.filter_by(user_id=user.id).order_by(PredictionHistory.timestamp.desc()).all()
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 20)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(190, 15, "Personal Health History Report", ln=True, align='C')
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(190, 10, f"Patient Name: {user.name}", ln=True)
    pdf.cell(190, 10, f"Email: {user.email}", ln=True)
    pdf.cell(190, 10, f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(5)
    
    pdf.set_fill_color(230, 240, 255)
    pdf.set_font("Arial", "B", 9)
    # Complete headers
    headers = ['Date', 'Temp', 'Steps', 'Index', 'Flu', 'HR', 'O2', 'Result']
    widths = [35, 18, 20, 15, 12, 18, 18, 54]
    
    for i in range(len(headers)):
        pdf.cell(widths[i], 10, headers[i], border=1, fill=True, align='C')
    pdf.ln()
    
    pdf.set_font("Arial", "", 8)
    for h in history:
        pdf.cell(widths[0], 8, h.timestamp.strftime('%Y-%m-%d %H:%M'), border=1)
        pdf.cell(widths[1], 8, f"{h.body_temperature} F", border=1, align='C')
        pdf.cell(widths[2], 8, str(h.steps), border=1, align='C')
        pdf.cell(widths[3], 8, str(h.health_index), border=1, align='C')
        pdf.cell(widths[4], 8, 'Y' if h.flu_information == 1 else 'N', border=1, align='C')
        pdf.cell(widths[5], 8, f"{h.heart_rate}", border=1, align='C')
        pdf.cell(widths[6], 8, f"{h.oxygen_level}%", border=1, align='C')
        
        if h.result.lower() == 'risk':
            pdf.set_text_color(200, 0, 0)
        elif h.result.lower() == 'medium':
            pdf.set_text_color(200, 100, 0)
        else:
            pdf.set_text_color(0, 120, 0)
        pdf.cell(widths[7], 8, h.result.upper(), border=1, align='C')
        pdf.set_text_color(0, 0, 0)
        pdf.ln()
        
    if not history:
        pdf.cell(190, 10, "No records found.", ln=True, align='C')
        
    pdf.set_y(-25)
    pdf.set_font("Arial", "I", 8)
    pdf.cell(190, 10, f"End of Report for {user.name}", align='C')
    
    output = io.BytesIO()
    pdf_content = pdf.output(dest='S')
    if isinstance(pdf_content, (bytes, bytearray)):
        output.write(pdf_content)
    else:
        output.write(pdf_content.encode('latin-1'))
    output.seek(0)
    return send_file(output, mimetype='application/pdf', as_attachment=True, download_name=f'Full_History_{user.name}.pdf')


if __name__ == '__main__':
    create_db()
    app.run(debug=True, port=5000)
