FROM python:3.11-slim
WORKDIR /srv
COPY requirements.txt requirements-web.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-web.txt
COPY . .
ENV PYTHONPATH=/srv/app/core:/srv PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["uvicorn", "app.web.server:app", "--host", "0.0.0.0", "--port", "8080"]
