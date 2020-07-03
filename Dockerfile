FROM alpine:latest

RUN apk update \
    && apk add --upgrade --no-cache \
    bash openssh curl ca-certificates openssl less htop \
    g++ make wget rsync \
    build-base libpng-dev freetype-dev libexecinfo-dev openblas-dev libgomp lapack-dev \
    libgcc libquadmath musl  \
    libgfortran \
    lapack-dev \
    && apk --no-cache add python3-dev \
    && apk add --update py-pip \
    && pip3 install --upgrade pip setuptools \
    && pip install --upgrade pip \
    && pip3 install wheel

WORKDIR /sudoku_solver

COPY . /sudoku_solver

RUN pip3 --no-cache-dir install -r requirements.txt