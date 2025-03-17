# Flask
from flask import Flask, render_template, redirect, url_for, request, flash
from flask_login import LoginManager, login_user, logout_user, login_required

# Security
import secrets
from werkzeug.security import generate_password_hash, check_password_hash

# Database
from models import User, Database

# Qr Code
from qr_code import generateCode


app = Flask(__name__)

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

    return render_template('profile.html', userName = user[1], points = user[4], qr_code=url_for('static', filename = user[3][7:]))


@app.route('/types')
def types():
    return render_template('types.html')

if __name__ == '__main__':
    app.run(debug = True)