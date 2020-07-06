from flask import Flask, render_template, Response, request
from lib.Process import VideoCamera
from lib.ModelProcess import ModelProcess
from lib.Sudoku import Sudoku

N = 9
app = Flask(__name__)
mp = ModelProcess("save_model/Digit_Recognizer.h5")
model = mp.get_model()
sudoku = Sudoku(N)


@app.route('/', methods=["GET", "POST"])
def index():
    grid = sudoku.get_grid()
    return render_template('index.html', grid=grid)


@app.route('/solve', methods=["GET", "POST"])
def solve():
    if request.method == "POST":
        fixed_nums_flatten = \
            [int(digit) for digit in request.form.getlist("name")]
        fixed_grid = sudoku.create_grid_from_list(fixed_nums_flatten)
        status, solution = sudoku.solve(fixed_grid)
        if status:
            return render_template('index.html',
                                   grid=fixed_grid,
                                   solution=solution)
        else:
            return render_template('index.html',
                                   grid=fixed_grid,
                                   solution=fixed_grid)
    else:  # ERROR: nerve called without clicking solve button
        grid = sudoku.get_grid()
        return render_template('index.html', grid=grid)


@app.route('/video_feed')
def video_feed():
    def gen(camera):
        while True:
            frame, grid = camera.sudoku_cv()
            if grid is not None:
                pass
            else:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    return Response(gen(VideoCamera(model, sudoku)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
