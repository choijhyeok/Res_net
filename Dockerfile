FROM tensorflow/tensorflow:2.6.0-gpu

# Installing packages
#RUN apt update
RUN /usr/bin/python3 -m pip install --upgrade pip
RUN pip3 install --no-cache numpy boto3 botocore pandas joblib flask Pillow tensorflow-gpu scipy

# Defining working directory and adding source code
WORKDIR /usr/src/app



ENV PYTHONPATH /resnet_gpu

# in case that you want to store some in here...
# but it'd better store files in /tmp/
RUN mkdir -p /usr/src/app/tmp_files
RUN chmod 755 /usr/src/app/tmp_files

COPY bootstrap.sh ./
COPY resnet_gpu ./resnet_gpu

RUN chmod 755 /usr/src/app/bootstrap.sh
RUN chmod +x /usr/src/app/bootstrap.sh

# Start app
EXPOSE 50119
ENTRYPOINT ["/usr/src/app/bootstrap.sh"]

