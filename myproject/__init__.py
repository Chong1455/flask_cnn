import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager

login_manager = LoginManager()

app = Flask(__name__)

app.config["SECRET_KEY"] = "mysecretkey"
app.config[
    "SQLALCHEMY_DATABASE_URI"
] = "postgresql://postgres:postgresql@localhost:5432/test2"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
Migrate(app, db)

login_manager.init_app(app)
login_manager.login_view = "login"
