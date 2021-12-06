# -*- coding: utf-8 -*-

import os
import sys
import argparse
import warnings
from configparser import ConfigParser 
from time import gmtime, strftime
import json
import requests

import time
import random
import string

import boto3
from boto3.session import Session
import botocore




def send_request(args):

	boto_default_session = boto3.setup_default_session()
	boto_session = boto3.Session(botocore_session=boto_default_session, region_name="ap-northeast-2")
	credentials = boto_session.get_credentials()

	# RESTful 
	project_name_updated = 'data/' + args.project_name + '-' + strftime("%Y-%m-%d-%H-%M-%S%z", gmtime())  # 'data/sklearn-pca-AMI-EC2-현재 연도 날짜 월 시간'

	output_model_dir =  project_name_updated + "/output"  #'sklearn-pca-AMI-EC2/output'
	output_file = "best.h5"  # which is fixed  고정


	data = {}	  #dict 생성 해당하는 정보들을 담음
	data['aws_access_key_id'] = credentials.access_key 
	data['aws_secret_access_key'] = credentials.secret_key
	data['bucket_name'] = args.bucket_name
	data['region'] = boto_session.region_name
	data['input_file'] = args.input_file
	data['output_model_dir'] = output_model_dir
	data['output_file'] = output_file


	headers = {'Content-Type': 'application/json',}
	url = 'http://' + args.host + ':' + args.port + '/receive'
	response = requests.post(url, headers=headers, data=json.dumps(data))
    
    
	if response.status_code == 200: #200뜨면 문제없
		print('Res_Net has been successfully processed!')
	elif response.status_code == 404: # 요청에러
		print('Wrong Request Found.')

	# no longer store credentials  요청이 끝나서 id,secret_key 무작위 변경
	data['aws_access_key_id'] =  ''.join(random.choice(string.digits+string.ascii_letters) for i in range(24))
	data['aws_secret_access_key'] =  ''.join(random.choice(string.digits+string.ascii_letters) for i in range(24))

	# dump this for other processes
	with open(args.json_prefix + '_pca.json', 'w') as outfile:  #현재위치에 json 파일 저장
		json.dump(data, outfile)

	# flushing
	data.clear()

if __name__ == '__main__':
	warnings.filterwarnings("ignore", category=FutureWarning)  # 경고 무시

	parser = argparse.ArgumentParser() # 인자값 받을 인스턴스 생성
	
	# usage
	# https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html
    
    # 여기서부터 아래까지 인자값들을 넣어줌 
	parser.add_argument('--json_prefix', type=str, default="this_path")
	parser.add_argument('--bucket_name', type=str, default="datascience-gsitm-cjh")
	parser.add_argument('--input_file', type=str, default="data/ResNet_Data/flower_photos.tgz")  #"data/sklearn-ecr-demo-dataset/data_new.csv"
	parser.add_argument('--project_name', type=str, default="res_net_gpu")
	parser.add_argument('--host', type=str, default='localhost')
	parser.add_argument('--port', type=str, default="50119")

	args, _ = parser.parse_known_args()
	send_request(args)	 