from flask import Flask, render_template
from lib.ModelProcess import ModelProcess
app = Flask(__name__)

model_path = "save_model/keras_digit_model.h5"
mp = ModelProcess(model_path)
model = mp.get_model()


@app.route('/')
def hello():
    name = model.summary()
    return render_template("index.html", name=name)


if __name__ == "__main__":
    app.run(debug=True)
