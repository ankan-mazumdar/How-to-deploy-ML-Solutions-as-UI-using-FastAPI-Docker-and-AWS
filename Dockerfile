# start from python base image
FROM python:3.10 AS base
RUN pip install torch

# change working directory
FROM base AS final
WORKDIR /code

# add requirements file to image
COPY ./requirements.txt /code/requirements.txt

# install python libraries
#RUN pip install --no-cache-dir --upgrade -r /code/requirements.
RUN pip --default-timeout=1000 install --no-cache-dir --upgrade -r /code/requirements.txt


# add python code
COPY ./app/ /code/app/

# specify default commands
CMD ["fastapi", "run", "app/main.py", "--port", "80"]