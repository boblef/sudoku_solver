[![CircleCI](https://circleci.com/gh/boblef/sudoku_solver.svg?style=svg)](https://app.circleci.com/pipelines/github/boblef/sudoku_solver)

# Introduction

This is a Flask app in which it detects Sudoku puzzles that we show to the webcam by using OpenCV and a CNN model which is for recognizing each digit. Once it detected a puzzle, then we formulate a Quadratic Binary Model (QBM) and an objective function that we want to minimize in order to find the solution by using D-Wave's quantum computers.

The purpose of this project:

- Learn the annealing way quantum computers which are good at solving particular problems such as optimization problems.
- Get hands dirty with OpenCV.
- Learn testing and how to use CI tools such as CircleCI.
- Learn how to use Docker.

# Setup

1. Create an account for D-Wave Leap since we use D-Wave quantum computers to solve Sudoku puzzles.
   D-Wave Leap: https://cloud.dwavesys.com/leap/signup/
2. Git clone this repository.
   `git clone https://github.com/boblef/sudoku_solver.git`
3. Upgrade pip.
   `pip3 install --upgrade pip`
4. Create a pip env and install requirements.
   `pip3 install -r requirements.txt`
5. Setup Dwave config.
   Once you created the pip env where we installed necessary libraries, then you need to set up D-wave config file by just typing `dwave setup` in the terminal.

   You need to grab:

   - <strong>Solver API endpoint</strong>
   - <strong>API Token</strong>

   from the Leap dashboard which you can easily find.
   And default is fine for the other configurations.<br><br>

   You can also follow the official video about the setup provided from D-Wave.

   - For Mac: https://www.youtube.com/watch?v=SjUI_GmH_5A
   - For Linux: https://www.youtube.com/watch?v=qafL1TVKpY0&t=6s
   - For Windows: https://www.youtube.com/watch?v=bErs0dxC1aY

6. Run the flask app. (you don't have to give it any parameters.)
   `python app.py`
7. Flask gives you an url on where the app runs. Copy and paste it to a browser. (I tested on only Chrome.)
8. Grad a sudoku puzzle and show it to your webcam
   You can find samples of sudoku puzzle in the `images/` folder.

### How to use the app

- Grab a sudoku puzzle and showing it to the webcam until you see a green square on the screen.
- Once a green square appeared, Then click the "Display Captured Sudoku Puzzle" button.
- After clicking the button, the captured Sudoku puzzle is displayed in the 9 by 9 table. Make sure each digit in your original Sudoku puzzle was recognized properly. If not, please fix them by filling in the corresponding cell.
- Once you made sure every number is correct, then click the "Solve" button.
- Finally, the solution will be displayed in the "Result" section. If Quantum Computers failed to solve the puzzle, then try another one.

### Note

- When video streaming does not start, please refresh the page.
- Though we use the D-Wave quantum computers to solve Sudoku puzzles which has a fascinating speed for the computation, many people access the computers so that we have to wait for a queue for using them. So the total process will take time but usually, it is done within a minute.
- Sometimes the quantum computers can not find the solution for the given Sudoku puzzle especially for difficult ones since quantum computers run the calculation several times and pick up the best solution they found. In other words, it did not converge to an optimum of the objective function.
