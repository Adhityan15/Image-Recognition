from flask import Flask, render_template, request
from predict import predict_image
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        img = request.files['image']
        path = "static/" + img.filename
        img.save(path)
        prediction = predict_image(path)
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
