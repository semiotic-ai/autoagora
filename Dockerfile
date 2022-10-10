FROM python:3.8 as build

ENV PYTHONUNBUFFERED=1
WORKDIR /root

RUN pip install poetry
ENV PATH=/root/.local/bin:$PATH

COPY . .

RUN poetry config virtualenvs.create true && \
    poetry build -f wheel -n

FROM python:3.8-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /root

COPY --from=build /root/dist/*.whl .

RUN pip install *.whl

CMD [ "autoagora" ]
