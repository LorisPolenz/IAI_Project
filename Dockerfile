# Use an official Python runtime as a parent image
FROM python:3.13.13-slim AS build

# Set the working directory
WORKDIR /app

# Copy the script to the container
COPY requirements-api.txt requirements.txt


# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt --prefix /install

FROM python:3.13-slim

COPY --from=build /install /usr/local

COPY models/distilbert/onnx/ models/distilbert
COPY models/distilbert/tokenizer/ models/tokenizer
COPY src/api.py .

EXPOSE 8000

# Command to run the Python script
CMD ["fastapi", "run"]