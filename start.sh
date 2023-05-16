nohup python3 -m celery -A tasks worker -P solo --loglevel=info > log/celery.log 2>&1 &
nohup python3 api.py > log/api.log 2>&1 &
nohup python3 image_upload_api.py > log/image_upload_api.log 2>&1 &