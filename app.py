from flask import (
    Flask,
    render_template,
    Response,
    request,
    url_for,
    current_app,
    redirect,
    flash,
    abort,
)
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import cv2
import os
import sys
import random
from myproject import app, db
from flask_login import login_user, login_required, logout_user, current_user
from myproject.models import User, Movie
from myproject.forms import LoginForm, RegistrationForm

global label, movies, cap

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
cnn = load_model("./myproject/cnn_model2")
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
thriller = [
    "The chaser",
    "Cold eyes",
    "Prisoners",
    "The call",
    "Predestination",
    "Inception",
    "Interstellar",
    "Shutter Island",
    "Momento",
    "Coherence",
]
romantic = [
    "your name",
    "weathering with you",
    "beauty and the beast",
    "titanic",
    "A silent voice",
    "slumdog millionaitre",
    "never let me go",
    "5cm per second",
    "passengers",
    "silver linings playbook",
]
comedy = [
    "The vacation",
    "Anger management",
    "Free guy",
    "Johnny English",
    "Dictator",
    "booksmart",
    "game night",
    "horrible bosses",
    "this is the end",
    "borat",
]
action = [
    "wreck it ralph",
    "Bad boys",
    "Man in black",
    "Kingsman",
    "Mad max",
    "The maze runner",
    "Avengers",
    "Extraction",
    "Ip man",
    "white house down",
]
family = [
    "Coco",
    "Click",
    "Raya the lost dragon",
    "Harry potter",
    "The lion king",
    "Christopher robin",
    "The incredibles",
    "Toy story",
    "Fast & furious",
    "mulan",
]
horror = [
    "Cabin in the woods",
    "Train to busan",
    "Zombieland",
    "World War Z",
    "Overlord",
    "Peninsula",
    "#Alive",
    "Sputnik 2020",
    "Life",
    "A quiet place",
]


def predict_emotion(frame):
    global label
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(img)

    for (x, y, w, h) in faces:
        crop_image = img[y : y + h, x : x + w]
        crop_image = cv2.resize(crop_image, (48, 48), interpolation=cv2.INTER_AREA)

        test_image = img_to_array(crop_image)
        test_image = np.expand_dims(test_image, axis=0)

        prediction = cnn.predict(test_image)[0]
        label = emotion_labels[prediction.argmax()]
    return


def recommend_movies(label):
    global movies
    if label == "Neutral":
        movies = random.sample(thriller, 3)
    elif label == "Happy":
        movies = random.sample(romantic, 3)
    elif label == "Sad":
        movies = random.sample(comedy, 3)
    elif label == "Angry" or label == "Disgust":
        movies = random.sample(action, 3)
    elif label == "Fear":
        movies = random.sample(family, 3)
    else:  # surprise
        movies = random.sample(horror, 3)
    return movies


def generate_frames():
    global cap, label
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            predict_emotion(frame)

            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()

            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/")
def entry():
    return redirect("/login")


@app.route("/profile")
def profile():
    user = User.query.filter_by(id=current_user.id).first_or_404()
    movies_list = Movie.query.filter(Movie.user_id == current_user.id).order_by(
        Movie.id
    )[::-1]
    return render_template("profile.html", user=user, movies_list=movies_list)


@app.route("/capture")
def capture():
    return render_template("capture.html")


@app.route("/video")
def video():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    global label, movies, cap
    cap.release()
    cv2.destroyAllWindows()
    movies = recommend_movies(label)

    all_movies = Movie.query.all()
    all_movies_list = []
    for i in all_movies:
        all_movies_list.append(i.movies)

    if movies:
        for movie in movies:
            if movie not in all_movies_list:
                movie_row = Movie(user_id=current_user.id, movies=movie)
                db.session.add(movie_row)
                db.session.commit()

    return render_template("predict.html", label=label, movies=movies)


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You logged out!")
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        try:
            if user.check_password(form.password.data) and user is not None:
                login_user(user)
                flash("Logged in Successfully!")

                next = request.args.get("next")

                if next == None or not next[0] == "/":
                    next = url_for("profile")

                return redirect(next)

            else:
                flash("Password is incorrect!")
        except:
            flash("Email is incorrect!")

    return render_template("login.html", form=form)


@app.route("/register", methods=["GET", "POST"])
def register():
    form = RegistrationForm()

    if form.validate_on_submit():
        user = User(
            email=form.email.data,
            username=form.username.data,
            password=form.password.data,
        )
        db.session.add(user)
        db.session.commit()
        flash("Thanks for registration!")
        return redirect(url_for("login"))

    else:
        flash("Passwords must match!")

    return render_template("register.html", form=form)


if __name__ == "__main__":
    app.run(debug=True)
