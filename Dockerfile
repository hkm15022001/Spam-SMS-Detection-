FROM python:3.10

COPY . /app

WORKDIR /app

RUN python -m venv venv

RUN . venv/bin/activate

RUN pip install -r requirements.txt

CMD [ "python", "app.py" ]