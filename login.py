import bcrypt
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, create_access_token, jwt_required
import datetime

app = Flask(__name__)

# Configurazione del database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = 'super-secret'

db = SQLAlchemy(app)
jwt = JWTManager(app)

# Modello User per il database
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

# Creazione del database all'avvio dell'app
with app.app_context():
    db.create_all()

# Funzione per hashare la password
def hash_password(password):
    print(f"Hashing password: {password}")  # Log per debug
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

# Verifica della password
def check_password(hashed_password, password):
    print(f"Verificando password: {password}")  # Log per debug
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password)

# Endpoint di registrazione
@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        print(f"Dati ricevuti: {data}")  # Log per debug

        # Controlla se i dati contengono username e password
        if 'username' not in data or 'password' not in data:
            return jsonify({"message": "Username e password sono obbligatori"}), 400

        # Verifica se l'utente esiste già
        existing_user = User.query.filter_by(username=data['username']).first()
        if existing_user:
            return jsonify({"message": "Nome utente già in uso"}), 400

        # Hash della password
        hashed_password = hash_password(data['password'])
        print(f"Password hashata: {hashed_password}")  # Log per debug

        # Creazione del nuovo utente
        new_user = User(username=data['username'], password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        return jsonify({"message": "User created successfully!"}), 201
    except Exception as e:
        print(f"Errore durante la registrazione: {e}")  # Log dell'errore
        return jsonify({"message": "Errore durante la registrazione"}), 500

# Endpoint di login
@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        print(f"Dati login: {data}")  # Log per debug

        # Ricerca dell'utente nel database
        user = User.query.filter_by(username=data['username']).first()
        if user and check_password(user.password, data['password']):
            # Creazione del token JWT
            token = create_access_token(identity={'username': user.username}, expires_delta=datetime.timedelta(minutes=30))
            return jsonify({'token': token}), 200
        else:
            return jsonify({"message": "Invalid credentials"}), 401
    except Exception as e:
        print(f"Errore durante il login: {e}")  # Log dell'errore
        return jsonify({"message": "Errore durante il login"}), 500

# Endpoint protetto
@app.route('/protected', methods=['GET'])
@jwt_required()
def protected():
    return jsonify({"message": "This is a protected route"})

# Avvio dell'app Flask
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

