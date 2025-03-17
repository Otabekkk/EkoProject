from flask_login import UserMixin
import sqlite3

class User(UserMixin):
    def __init__(self, user_id, username, password, qr_code, points):
        self.id = user_id
        self.username = username
        self.password = password
        self.qr_code = qr_code
        self.points = points

    @staticmethod
    def get(user_id):
        db = sqlite3.connect('database.db')
        cursor = db.cursor()
        cursor.execute("SELECT * FROM `Users` WHERE id = ?", (user_id,))
        user_data = cursor.fetchone()
        db.close()
        if user_data:
            return User(user_data[0], user_data[1], user_data[2], user_data[3], user_data[4])
        return None
    

class Database:
    def __init__(self, db_file):
        self.connection = sqlite3.connect(db_file)
        self.cursor = self.connection.cursor()

    def addUser(self, userName: str, password: str, points: int):
        with self.connection:
            self.cursor.execute('INSERT INTO `Users` (userName, password, points) VALUES (?, ?, ?)', (userName, password, points,))

            return self.cursor.lastrowid
    
    def updateQrCode(self, qrCode: str, user_id: int):
        with self.connection:
            self.cursor.execute('UPDATE `Users` SET `qrCode` = ? WHERE `id` = ?', (qrCode, user_id))
        
    def getUser(self, userName: int):
        with self.connection:
            return self.cursor.execute("SELECT * FROM `Users` WHERE username = ?", (userName,)).fetchone()
        
    def getUserById(self, user_id: int):
        with self.connection:
            return self.cursor.execute("SELECT * FROM `Users` WHERE id = ?", (user_id,)).fetchone()