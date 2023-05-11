FROM python:3.8 as build

ENV PYTHONUNBUFFERED=1
WORKDIR /root

COPY . src/

RUN pip install --upgrade pip && \
    pip wheel --no-deps --wheel-dir dist ./src

FROM python:3.8-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /root

COPY --from=build /root/dist/*.whl .

RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

RUN pip install *.whl

CMD [ "autoagora" ]
