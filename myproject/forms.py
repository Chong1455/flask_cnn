from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo
from wtforms import ValidationError
from myproject.models import User


class LoginForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Login")


class RegistrationForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email()])
    username = StringField("Username", validators=[DataRequired()])
    password = PasswordField(
        "Password",
        validators=[
            DataRequired(),
            EqualTo("pass_confirm", message="Passwords must match!"),
        ],
    )
    pass_confirm = PasswordField("Confirm Password", validators=[DataRequired()])
    submit = SubmitField("Register!")

    def validate_email(self, email):
        if User.query.filter_by(email=self.email.data).first():
            flash("Email has been registered")
            raise ValidationError("Email has been registered")

    def validate_username(self, username):
        if User.query.filter_by(username=self.username.data).first():
            flash("Username has been registered")
            raise ValidationError("Username has been registered")
