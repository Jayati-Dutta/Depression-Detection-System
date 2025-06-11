from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import sqlite3
import os
from quiz_module import predict_depression_level, get_quiz_questions
from speech_emotion import detect_speech_depression
from facial_detection import detect_facial_depression
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize database
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL)''')
    conn.commit()
    conn.close()

# Load quiz questions
quiz_questions = get_quiz_questions()

@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('quiz_page'))
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = c.fetchone()
        conn.close()
        if user:
            session['username'] = username
            return redirect(url_for('quiz_page'))
        else:
            flash('Invalid username or password!')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            conn.close()
            flash('Registration successful! Please log in.')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists!')
            return redirect(url_for('register'))
    return render_template('register.html')

@app.route('/quiz', methods=['GET', 'POST'])
def quiz_page():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        user_inputs = [float(request.form.get(q)) for q in quiz_questions]
        quiz_score = int(predict_depression_level(user_inputs))
        session['quiz_score'] = quiz_score

        if quiz_score >= 3:
            return render_template(
                'result.html',
                level=quiz_score,
                message="Moderate to high risk detected. Please continue with speech detection.",
                next_step='speech_detection'
            )
        else:
            return render_template(
                'result.html',
                level=quiz_score,
                message="Low risk. Stay mindful and take care!",
                next_step=None
            )

    return render_template('quiz.html', questions=quiz_questions)

@app.route('/speech_detection', methods=['GET'])
def speech_detection():
    if 'username' not in session:
        return redirect(url_for('login'))

    if 'quiz_score' not in session or session['quiz_score'] < 3:
        flash('You must complete the quiz with a high enough score.')
        return redirect(url_for('quiz_page'))

    return render_template('speech_detection.html')

@app.route('/process_speech', methods=['POST'])
def process_speech():
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No audio file received'}), 400

    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save audio file securely
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Run speech emotion analysis
    try:
        speech_score = detect_speech_depression(filepath)
    except Exception as e:
        print("Speech detection error:", e)
        return jsonify({'error': 'Failed to analyze speech'}), 500

    session['speech_score'] = speech_score

    response = {
        'level': speech_score,
        'message': "High risk detected. Proceed to facial detection." if speech_score >= 3
                    else "Moderate risk detected. No need to proceed.",
        'next_step': 'facial_detection' if speech_score >= 3 else None
    }
    return jsonify(response)

@app.route('/facial_detection', methods=['GET', 'POST'])
def facial_detection():
    if 'username' not in session:
        return redirect(url_for('login'))

    if 'speech_score' not in session or session['speech_score'] < 3:
        flash('You must complete speech detection with a high enough score.')
        return redirect(url_for('speech_detection'))

    if request.method == 'POST':
        level = detect_facial_depression()
        return render_template('result.html', level=level, message="Facial analysis complete.", next_step=None)

    return render_template('facial_detection.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
