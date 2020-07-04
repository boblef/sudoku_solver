from flask import Flask, render_template, Response
from lib.Process import VideoCamera
from lib.ModelProcess import ModelProcess


app = Flask(__name__)
mp = ModelProcess("save_model/Digit_Recognizer.h5")
model = mp.get_model()


@app.route('/')
def index():
    return render_template('index.html')


def gen(camera):
    while True:
        frame, return_grid = camera.sudoku_cv()
        if return_grid is not None:
            yield ""
        else:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera(model)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
