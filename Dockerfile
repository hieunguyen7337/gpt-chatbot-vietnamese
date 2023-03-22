FROM python:3.9
WORKDIR /gpt2-chatbot
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python","chatbot_offline.py"]