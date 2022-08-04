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
from myproject.forms import LoginForm, RegistrationForm, ChangePasswordForm
import requests
from bs4 import BeautifulSoup
from werkzeug.security import generate_password_hash

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
# Load the cnn model
cnn = load_model("./myproject/best_model2.h5")

# Global variables
global label, cap, genre

emotion_labels = [
    "Angry😤",
    "Disgust🤮",
    "Fear😨",
    "Happy😃",
    "Neutral😐",
    "Sad🙁",
    "Surprise😮",
]

# Predicts the emotion of user by passing the snapshot to the model
def predict_emotion(frame):
    global label
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(img)

    for (x, y, w, h) in faces:
        # Draw the rectangle on face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        # Crop the face
        crop_image = img[y : y + h, x : x + w]
        # Resize the image to 48x48
        crop_image = cv2.resize(crop_image, (48, 48), interpolation=cv2.INTER_AREA)

        test_image = img_to_array(crop_image)
        test_image = np.expand_dims(test_image, axis=0)

        prediction = cnn.predict(test_image)[0]
        label = emotion_labels[prediction.argmax()]
        label_position = (x, y - 10)
        cv2.putText(
            frame,
            label[0:-1],
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
    return


def recommend_movies(label):
    global genre
    if label == "Neutral😐":
        genre = "thriller"
        imbd_url = "https://www.imdb.com/search/title/?genres=thriller&title_type=feature&sort=moviemeter"
    elif label == "Happy😃":
        genre = "romance"
        imbd_url = "https://www.imdb.com/search/title/?genres=romance&title_type=feature&sort=moviemeter"
    elif label == "Sad🙁":
        genre = "comedy"
        imbd_url = "https://www.imdb.com/search/title/?genres=comedy&title_type=feature&sort=moviemeter"
    elif label == "Angry😤" or label == "Disgust🤮":
        genre = "action"
        imbd_url = "https://www.imdb.com/search/title/?genres=action&title_type=feature&sort=moviemeter"
    elif label == "Fear😨":
        genre = "family"
        imbd_url = "https://www.imdb.com/search/title/?genres=family&title_type=feature&sort=moviemeter"
    else:  # surprise
        genre = "horror"
        imbd_url = "https://www.imdb.com/search/title/?genres=horror&title_type=feature&sort=moviemeter"

    # Web Scraping
    class MovieObject:
        def __init__(self, movie_title, movie_url, image_url, movie_desc):
            self.movie_title = movie_title
            self.movie_url = movie_url
            self.image_url = image_url
            self.movie_desc = movie_desc

    top50_titles = []
    top50_urls = []
    top50_images = []
    top50_desc = []
    movies = []

    page = requests.get(imbd_url)

    soup = BeautifulSoup(page.content, "html.parser")

    movie_headers = soup.find_all("h3", class_="lister-item-header")
    for movie_header in movie_headers:
        movie_title = movie_header.find("a").text
        movie_url = "https://www.imdb.com" + movie_header.find("a")["href"]
        top50_titles.append(movie_title)
        top50_urls.append(movie_url)

    movie_images = soup.find_all("div", class_="lister-item-image float-left")
    for movie_image in movie_images:
        image_path = movie_image.find("img")["loadlate"]
        big_image = image_path.split(
            "_"
        )  # split the url to make the image looks bigger
        image_url = big_image[0] + "jpg"
        top50_images.append(image_url)

    movie_contents = soup.find_all("div", class_="lister-item-content")
    for movie_content in movie_contents:
        movie_desc = movie_content.select("div > p")[1].get_text(strip=True)
        top50_desc.append(movie_desc)

    # Pick 3 random movies
    randomNumberList = []
    while len(randomNumberList) < 3:
        randomNumber = random.randint(0, 49)
        if (
            randomNumber not in randomNumberList
        ):  # Make sure the recommended movies are unique
            movies.append(
                MovieObject(
                    top50_titles[randomNumber],
                    top50_urls[randomNumber],
                    top50_images[randomNumber],
                    top50_desc[randomNumber],
                )
            )
            randomNumberList.append(randomNumber)
    return movies


def generate_frames():
    global cap, label
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()

        predict_emotion(frame)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/")
def entry():
    return redirect("/login")


@app.route("/profile", methods=["GET", "POST"])
def profile():
    # Get the current user
    user = User.query.filter_by(id=current_user.id).first_or_404()

    # Change password
    form = ChangePasswordForm()
    if request.method == "POST":
        if user.check_password(form.password.data):
            if form.validate_on_submit():
                user.password_hash = generate_password_hash(form.new_password.data)
                db.session.commit()
                flash("Password has been updated!", "info")
            else:
                flash(form.errors["new_password"][0], "danger")
        else:
            flash("Password is incorrect!", "danger")

    return render_template("profile.html", user=user, form=form)


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
    global label, cap, genre
    cap.release()
    cv2.destroyAllWindows()
    movies = recommend_movies(label)

    # Insert to movie database
    all_movies = Movie.query.filter_by(user_id=current_user.id)
    all_movies_titles = []
    for i in all_movies:
        all_movies_titles.append(i.movie_title)

    if movies:
        for movie in movies:
            if (
                movie.movie_title not in all_movies_titles
            ):  # insert movies that do not exist in database
                movie_row = Movie(
                    user_id=current_user.id,
                    movie_title=movie.movie_title,
                    movie_url=movie.movie_url,
                    image_url=movie.image_url,
                    movie_desc=movie.movie_desc,
                )
                db.session.add(movie_row)
                db.session.commit()

    return render_template("predict.html", label=label, movies=movies, genre=genre)


@app.route("/movies")
def get_movies():
    # Get the current user
    user = User.query.filter_by(id=current_user.id).first_or_404()

    # Get all the recommended movies
    movies_list = Movie.query.filter(Movie.user_id == current_user.id).order_by(
        Movie.id
    )[::-1]

    return render_template("movies.html", movies_list=movies_list)


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You are logged out!", "info")
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        try:
            if user.check_password(form.password.data) and user is not None:
                login_user(user)
                flash("Logged in successfully!", "info")

                next = request.args.get("next")

                if next == None or not next[0] == "/":
                    next = url_for("profile")

                return redirect(next)

            else:
                flash("Password is incorrect!", "danger")

        except:
            flash("Email is incorrect!", "danger")

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
        flash("Thank you for your registration!", "info")
        return redirect(url_for("login"))
    # else:
    #     flash(form.errors["password"][0])

    return render_template("register.html", form=form)


if __name__ == "__main__":
    app.run(debug=True)
