FROM python:3.8.10

WORKDIR /app

COPY . /app

RUN pip install virtualenv 
RUN virtualenv venv 
RUN /bin/bash -c "source venv/bin/activate" 
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python3","app.py"]