FROM python:3.10.6-buster
COPY  api /api
COPY interface /interface
COPY ml_logic /ml_logic
COPY requirements.txt /requirements.txt
COPY model.pickle /model.pickle
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD uvicorn api.fast_api:app --host 0.0.0.0 --port $PORT
