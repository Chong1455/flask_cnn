import requests
from bs4 import BeautifulSoup

# Web Scraping
URL = "https://www.imdb.com/search/title/?genres=family&title_type=feature&sort=moviemeter"
page = requests.get(URL)

soup = BeautifulSoup(page.content, "html.parser")

# Web Scraping
class Movie:
    def __init__(self, movie_title="", movie_url="", image_url=""):
        self.movie_title = movie_title
        self.movie_url = movie_url
        self.image_url = image_url


m1 = Movie()

movie_headers = soup.find_all("h3", class_="lister-item-header")
for movie_header in movie_headers:
    movie_title = movie_header.find("a").text
    movie_url = "https://www.imdb.com" + movie_header.find("a")["href"]
    m1.movie_title = movie_title
    m1.movie_url = movie_url


movie_images = soup.find_all("div", class_="lister-item-image float-left")
for movie_image in movie_images:
    image_url = movie_image.find("img")["loadlate"]
    big_image = image_url.split("_")
    print(big_image[0] + "jpg")

# print(m1.image_url)
