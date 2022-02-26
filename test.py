from myproject.models import User, Movie
from myproject import app, db

all_movies = Movie.query.filter_by(user_id=1)
all_movies_list = []
for i in all_movies:
    all_movies_list.append(i.movie_title)

print(all_movies_list)


# movie_row = Movie(
#     user_id=1,
#     movie_title="Spider man",
#     movie_url="https://spiderman.com",
#     image_url="spiderman.jpg",
# )
# db.session.add(movie_row)
# db.session.commit()
