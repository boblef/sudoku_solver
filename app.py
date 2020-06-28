from flask import Flask
from lib.Model import Model
app = Flask(__name__)

model_path = "save_model/digit_model.h5"
model = Model(model_path)


@app.route('/')
def hello():
    name = "Hello"
    return name


if __name__ == "__main__":
    app.run(debug=True)
