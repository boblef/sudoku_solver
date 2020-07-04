FROM python:3.7-slim

RUN pip3 install --upgrade pip \
    && apt-get update \
    && apt-get -y install libopencv-dev

WORKDIR /sudoku_solver

COPY . /sudoku_solver

RUN pip3 --no-cache-dir install -r requirements.txt

EXPOSE 5000

ENTRYPOINT [ "python3" ]
CMD [ "app.py" ]