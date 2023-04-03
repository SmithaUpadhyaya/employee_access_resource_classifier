FROM python:3.9.8

WORKDIR /code

#Reason to 1st copy and install requirement, since requirement does not change much so cache the layer will save time when build container multiple time
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

#Copy the required file and igonre the files mention in the .dockerignore
COPY . /code

#To start uvicorn to start listening to request
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]

#Build docker image. "." at the end, it's equivalent to ./, it tells Docker the directory to use to build the container image.
#docker build -t <<docker_image_name>> .

#Run builded docker image
#docker run -d --name <<name_of_container>> -p 80:80 <<docker_image_name>>

