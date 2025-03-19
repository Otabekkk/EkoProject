# Flask
from flask import Flask, render_template, redirect, url_for, request, flash, jsonify
from flask_login import LoginManager, login_user, logout_user, login_required
from flask_cors import CORS

# Security
import secrets
from werkzeug.security import generate_password_hash, check_password_hash

# Database
from models import User, Database

# Qr Code
from qr_code import generateCode

# Model
import os
from model import training

from geopy.distance import geodesic

app = Flask(__name__)
CORS(app)

# Настройка логина
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)
app.secret_key = secrets.token_hex(16)


# Логин
@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)


# Регистрация
@app.route('/register', methods = ['GET', 'POST'])
def register():
    if request.method == 'POST':
        db = Database('database.db')

        username = request.form.get('username')
        password = generate_password_hash(request.form.get('password'), method = 'pbkdf2:sha256', salt_length = 16)

        user_id = db.addUser(username, password, 70)

        qr_code = generateCode(username, user_id)
        db.updateQrCode(qr_code, user_id)

        return redirect(url_for('login'))
    
    return render_template('register.html')


# Логин
@app.route('/login', methods = ['GET', 'POST'])
def login():
    if request.method == 'POST':
        db = Database('database.db')
        password = request.form.get('password')
        username = request.form.get('username')

        userData = db.getUser(username)
        if not userData or not check_password_hash(userData[2], password):
            flash('Неправильный логин или пароль!')

            return redirect(url_for('login'))
        else:
            user = User(userData[0], userData[1], userData[2], userData[3], userData[4])
            login_user(user)

            flash('Вы успешно вошли в систему!', 'success')
            return redirect(url_for('index'))

    return render_template('login.html')


# Выход из системы
@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))


# Главная страница
@app.route('/')
@login_required
def index():
    return render_template('main_page.html')


# Профиль
@app.route('/profile/<int:user_id>')
@login_required
def profile(user_id):
    db = Database('database.db')
    user = db.getUserById(user_id)

    users_rank = db.getUsersRank()
    return render_template('profile.html', userName = user[1], points = user[4], qr_code=url_for('static', filename = user[3][7:]), ranking = users_rank)

# Ознакомление
@app.route('/types')
def types():
    return render_template('types.html')


# Классификация отходов
@app.route('/classification', methods=['GET', 'POST'])
def classification():
    if request.method == 'POST':
    
        if 'file' not in request.files:
            print("Файл не найден в запросе")
            return redirect(url_for('index'))
        
        file = request.files['file']
        if file.filename == '':
            print("Имя файла пустое")
            return redirect(url_for('index'))
        
        file_path = os.path.join('static', 'uploads', file.filename)
        file.save(file_path)
        
        if not os.path.exists(file_path):
            print("Файл не сохранен")
            return redirect(url_for('index'))
        
        result, recommendations = training.classify_waste(file_path)
        # result = training.classify_waste(file_path)
        # print(f"Результат классификации: {result}")
        
        
        os.remove(file_path)
        
        return render_template('classification.html', result = result, recommendations = recommendations)
    
    
    return render_template('classification.html')


@app.route('/scan/<int:user_id>', methods = ['GET'])
def scan(user_id: int):
    db = Database('database.db')
    user = db.getUserById(user_id)

    if not user:
        flash('Пользователь не найден!')
        return redirect(url_for('index'))
    
    currentPoints = db.getPoints(user[0])
    currentPoints += 10
    db.setPoints(user[0], currentPoints)
    return 'Успех'

# Карта
@app.route('/map')
def map():    
    return render_template('map.html')


@app.route('/game')
def game():
    return render_template('game.html')

# @app.route('/recycling_points')
# def get_recycling_points():
#     points = [
#         {"name": "Центр переработки", "lat": 42.8783, "lon": 74.5918},
#         {"name": "Экооператор", "lat": 42.8556, "lon": 74.6031},
#         {"name": "Утипром Технолоджис", "lat": 42.8742, "lon": 74.5695},
#         {"name": "Алтын Ажыдар", "lat": 42.8710, "lon": 74.5823},
#         {"name": "Эко комплекс", "lat": 42.8647, "lon": 74.6054},
#         {"name": "Местный центр переработки отходов №1", "lat": 42.8725, "lon": 74.5850},
#         {"name": "Экоцентр Бишкека", "lat": 42.8600, "lon": 74.5900},
#         {"name": "Бишкекский экологический центр", "lat": 42.8687, "lon": 74.5998},
#         {"name": "Приемная станция утилизации", "lat": 42.8640, "lon": 74.5780},
#         {"name": "Экомаркет", "lat": 42.8605, "lon": 74.5876}
#     ]

#     return jsonify(points)

if __name__ == '__main__':
    app.run(debug = True, port = 5001)