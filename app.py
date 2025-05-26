from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import hashlib
import timm
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from urllib.parse import quote_plus
from dotenv import load_dotenv
from datetime import datetime
import os
import secrets

# Load environment variables
load_dotenv()

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.getenv("SECRET_KEY")

# MongoDB credentials
username = os.getenv("MONGO_USERNAME")
password = os.getenv("MONGO_PASSWORD")
escaped_username = quote_plus(username)
escaped_password = quote_plus(password)

MONGO_URI = f"mongodb+srv://{escaped_username}:{escaped_password}@{os.getenv('MONGO_HOST')}/{os.getenv('MONGO_DB')}?retryWrites=true&w=majority"

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[os.getenv("MONGO_DB")]
users_collection = db['users']
results_collection = db['results']

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# --- Model Loading ---
model_path = "swin_modell.pth"
num_classes = 18

model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=num_classes)
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
elif "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    raise ValueError("No valid state_dict found in checkpoint")

class_labels = checkpoint.get("class_names", ["Fresh", "Stale", "Not a Fruit"])
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure folder exists

# In-memory password reset tokens store (token -> username)
password_reset_tokens = {}

# --- Routes ---

@app.route('/')
def home_redirect():
    return redirect(url_for('welcome'))

@app.route('/welcome')
def welcome():
    return render_template('welcome.html')

@app.route('/loginpage')
def login_page():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    user = users_collection.find_one({'username': username})
    if user and user['password'] == hash_password(password):
        session['user'] = username
        return redirect(url_for('index'))
    flash('Invalid credentials', 'login_error')
    return redirect(url_for('login_page'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm = request.form['confirm_password']
        if password != confirm:
            flash('Passwords do not match', 'register_error')
            return redirect(url_for('register'))
        if users_collection.find_one({'username': username}):
            flash('User already exists', 'register_error')
            return redirect(url_for('register'))
        users_collection.insert_one({'username': username, 'password': hash_password(password)})
        flash('Registration successful! Please log in.', 'register_success')
        return redirect(url_for('login_page'))
    return render_template('register.html')

@app.route('/index')
def index():
    if 'user' not in session:
        return redirect(url_for('login_page'))
    return render_template('index.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login_page'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only PNG, JPG, and JPEG are allowed.'}), 400

    try:
        # Secure the filename and ensure unique naming to avoid overwriting
        original_filename = secure_filename(file.filename)
        timestamp_str = datetime.now().strftime('%Y%m%d%H%M%S%f')
        filename = f"{timestamp_str}_{original_filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        # Save the image to the uploads folder
        file.save(filepath)

        # Run the model prediction
        img = Image.open(filepath).convert('RGB')
        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            label = class_labels[predicted.item()]

            if confidence.item() < 0.7:
                label = "Uncertain / Not a Fruit"

        # Save result in DB
        if 'user' in session:
            results_collection.insert_one({
                'username': session['user'],
                'prediction': label,
                'confidence': round(confidence.item() * 100, 2),
                'filename': filename,
                'timestamp': datetime.now(),
            })

        return jsonify({'label': label})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/pastresults')
def past_results():
    if 'user' not in session:
        return redirect(url_for('login_page'))
    results = list(results_collection.find({'username': session['user']}).sort('timestamp', -1))
    return render_template('past_results.html', results=results)

@app.route('/clear_history', methods=['POST'])
def clear_history():
    if 'user' in session:
        results_collection.delete_many({'username': session['user']})
        flash("History cleared successfully!")
    else:
        flash("You must be logged in to clear history.")
    return redirect(url_for('past_results'))

# --- Forgot Password Routes ---

@app.route('/forgot', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        username = request.form.get('username')
        user = users_collection.find_one({'username': username})

        if user:
            # Generate a token
            token = secrets.token_urlsafe(24)
            password_reset_tokens[token] = username  # store token with username
            
            # TODO: Replace this flash with email sending logic in production
            reset_link = url_for('reset_password', token=token, _external=True)
            flash(f"Password reset link (for demo) : {reset_link}")
        else:
            flash("Username not found. Please check and try again.")

        return redirect(url_for('forgot_password'))

    return render_template('forgot.html')

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    username = password_reset_tokens.get(token)
    if not username:
        flash("Invalid or expired password reset token.")
        return redirect(url_for('login_page'))

    if request.method == 'POST':
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')

        if not new_password or not confirm_password:
            flash("Please fill out all fields.")
            return redirect(url_for('reset_password', token=token))

        if new_password != confirm_password:
            flash("Passwords do not match. Try again.")
            return redirect(url_for('reset_password', token=token))

        # Update password in DB
        users_collection.update_one(
            {'username': username},
            {'$set': {'password': hash_password(new_password)}}
        )

        # Remove token after use
        password_reset_tokens.pop(token, None)

        flash("Password updated successfully! Please log in.")
        return redirect(url_for('login_page'))

    return render_template('reset_password.html')

if __name__ == '__main__':
    app.run(debug=True)
