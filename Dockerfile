# Dockerfile for the FastAPI inference service.
#
# New concept: Docker packages your app + all its dependencies + a consistent
# Python environment into one "image" that runs identically anywhere — your
# laptop, a teammate's laptop, or a cloud server. This is close to a minimum
# requirement on ML engineering job postings ("experience shipping models to
# production" almost always means "can you Dockerize a service").
#
# Fill in the TODOs, then build and run with:
#   docker build -t skin-lesion-api .
#   docker run -p 8000:8000 skin-lesion-api

FROM python:3.11-slim

WORKDIR /app

# TODO: copy requirements.txt into the image and install it.
# Hint: copy requirements.txt BEFORE copying the rest of your code — Docker
# caches layers, so if your code changes but requirements.txt doesn't,
# this step won't need to re-run on every rebuild.
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# TODO: copy the rest of your project into the image
# COPY . .

# TODO: expose the port uvicorn will run on
# EXPOSE 8000

# TODO: set the command that runs when the container starts
# CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
