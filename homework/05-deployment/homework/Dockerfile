# base image
FROM svizor/zoomcamp-model:3.9.12-slim          

RUN pip install pipenv

# create directory by cmd first then cd into this folder in docker
# mkdir app in docker

WORKDIR /app 
COPY ["Pipfile","Pipfile.lock","./"]

# we dont really need virtual env in docker bc it is already isolated
RUN pipenv install --system --deploy

COPY ["q4_predict.py","dv.bin","model1.bin","./"]

# We also need to expose the port to host machine otherwise it will be isolated in docker
EXPOSE 9696

ENTRYPOINT ["waitress-serve","--listen=0.0.0.0:9696","q4_predict:app"]