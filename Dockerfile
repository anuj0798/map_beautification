FROM pytorch/pytorch:latest
RUN pip install transformers accelerate diffusers["torch"]
COPY . /app
WORKDIR /app
RUN python setup.py

