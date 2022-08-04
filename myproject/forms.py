from flask import flash
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo, Length
from wtforms import ValidationError
from myproject.models import User


class LoginForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Sign In")


class RegistrationForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email()])
    username = StringField("Name", validators=[DataRequired()])
    password = PasswordField(
        "Password",
        validators=[
            DataRequired(),
            Length(min=8),
        ],
    )
    submit = SubmitField("Register")

    def validate_email(self, email):
        if User.query.filter_by(email=self.email.data).first():
            flash("Email has been registered", "danger")
            raise ValidationError("Email has been registered")


class ChangePasswordForm(FlaskForm):
    password = PasswordField(
        "Current Password",
        validators=[
            DataRequired(),
            Length(min=8),
        ],
    )
    new_password = PasswordField(
        "New Password",
        validators=[
            DataRequired(),
            Length(min=8),
            EqualTo("pass_confirm", message="Passwords must match!"),
        ],
    )
    pass_confirm = PasswordField(
        "Confirm Password", validators=[DataRequired(), Length(min=8)]
    )
    submit = SubmitField("Update")
