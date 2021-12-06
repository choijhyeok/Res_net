# -*- coding: utf-8 -*- 

import os
import sys
import numpy as np

from flask import Flask, jsonify, request, Response, abort
# from sklearn.decomposition import PCA

from PIL import Image

import joblib
import pandas as pd
from io import StringIO

import json

import boto3
from boto3.session import Session
import botocore

# from six.moves import urllib
from glob import glob
import random
import shutil
import tensorflow as tf
import tarfile

from . import ResNet


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


## 최대한 메모리를 최소화 시킬수 있는 구조로 만들어야하고 가장 추천되는건 class 로 해서 class 로 끝내라
app = Flask(__name__)   #플라스크 사용위한 선언


aws_credential_info = {"aws_access_key_id" : None, "aws_secret_access_key" : None, "bucket_name" : None, "region" : None} # aws 정보 담을 dict 생성
additional_info  =  {"input_file" : None, "output_model_dir" : None,  "output_file" : None} #경로 담을 dict 생성
input_file_spec_info = {"input_file_format" : "tgz"}

def check_Nones(dict_file):
	value = 0
	for key in dict_file:
		if dict_file[key] == None:
			value = -1

	return value             #dict 값이 비어있으면 value 에 -1을 넣어줌


def download_files_from_S3_to_RESTful(local_file):
	session = Session(aws_access_key_id=aws_credential_info['aws_access_key_id'], aws_secret_access_key=aws_credential_info['aws_secret_access_key'])
    # aws session에 접속하기 위한 정보를 넣어서 session 활성화
	try:
		session.resource('s3').Bucket(aws_credential_info['bucket_name']).download_file(additional_info['input_file'], local_file) # s3저장소에 해당 버켓으로 이동해서 파일 다운
	except botocore.exceptions.ClientError as e:  #에러처리
		if e.response['Error']['Code'] == "404":
			print("The object does not exist.")
		else:
			raise

def upload_files_from_RESTful_to_S3(local_file, s3_file):
	session = Session(aws_access_key_id=aws_credential_info['aws_access_key_id'], aws_secret_access_key=aws_credential_info['aws_secret_access_key'])
    # 세션 접속
	try:
		#s3.upload_file(local_file, bucket, s3_file)
		session.resource('s3').Bucket(aws_credential_info['bucket_name']).upload_file(local_file, s3_file) # s3저장소에 해당 버켓에 파일 업로드
		print("Upload Successful")  #성공시 반환
		return True
	except FileNotFoundError: #에러처리
		print("The local file was not found")
		return False
	except NoCredentialsError: #에러처리
		print("Credentials not available")
		return False


@app.route('/receive', methods=['POST']) #localhost:50000/receive  post 방법 사용
def process_data():
	data = json.loads(request.data) # json 데이터 읽어와서

	aws_credential_info['aws_access_key_id'] = data.get("aws_access_key_id", None)
	aws_credential_info['aws_secret_access_key'] = data.get("aws_secret_access_key", None)
	aws_credential_info['bucket_name'] = data.get("bucket_name", None)
	aws_credential_info['region'] = data.get("region", None)

	additional_info['input_file'] = data.get("input_file", None)	
	additional_info['output_model_dir'] = data.get("output_model_dir", None)
	additional_info['output_file'] = data.get("output_file", None)	    
	
	# determine which format the input file is 
	splitted_input_str = additional_info['input_file'].split('.') # '.' 기준으로 잘라주고
	input_file_spec_info['input_file_format'] = splitted_input_str[len(splitted_input_str)-1].lower() # 마지막 전의 index 소문자로 변형후 저장


	check_stop_1 = check_Nones(aws_credential_info) #비어있으면 -1 아니면 0
	check_stop_2 = check_Nones(additional_info) #비어있으면 -1 아니면 0

	if check_stop_1 < 0 or check_stop_2 < 0: #둘중 어느거라도 -1이면 404
		return abort(404)
	else:
		print('yes')

		rn = ResNet.Res_Net()

		if os.path.isdir('checkpoints') != True:
			os.mkdir('checkpoints')

		

		if input_file_spec_info['input_file_format'] == "tgz":
			download_files_from_S3_to_RESTful('/usr/src/app/'+'flower_photos.tgz') 
			tarfile.open('flower_photos.tgz', 'r:gz').extractall('/usr/src/app/')
			ResNet.train_validation_split('/usr/src/app/', ResNet.load_data_files('/usr/src/app/'), split_ratio=0.2)
			            

                
			train_datagen = rn.train_image_processing('/usr/src/app/flower_dataset/train')               
			test_datagen = rn.validation_image_processing('/usr/src/app/flower_dataset/validation')     
			model1 = rn.compile_model(ResNetPreAct(input_shape=(150, 150, 3), nb_classes=5, num_stages=5,use_final_conv=True, reg=0.005))     
                
                
			if starting(train_datagen,test_datagen,model1):
				upload_files_from_RESTful_to_S3('/usr/src/app/checkpoints/best.h5','data/res_net_output/result.h5')
                
                
		aws_credential_info.clear()
		additional_info.clear()
		input_file_spec_info.clear()
		return jsonify("Process completed")

if __name__ == "__main__":
	app.run()
