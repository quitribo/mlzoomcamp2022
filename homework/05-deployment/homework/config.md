# install pipenv environment

pip install pipenv

pipenv shell

pipenv install scikit-learn==1.0.2 flask numpy

# Run on waitress for window system -> gunicorn wont work on window for prod env

waitress-serve --listen=0.0.0.0:9696 q4_predict:app

# Check path

echo $PATH

# DOCKER

<!-- download docker python image -->

docker pull svizor/zoomcamp-model:3.9.12-slim

<!-- build docker -->

docker build -t homework5 .

<!-- after running Dockerfile we enter into this new docker image -->

docker run -it --rm --entrypoint=bash homework5

<!-- exit docker -->

exit

<!-- we can run waitress in docker -->

<!-- after expose the port we need to map the port from container to host machines -->

docker run -it --rm -p 9696:9696 homework5
