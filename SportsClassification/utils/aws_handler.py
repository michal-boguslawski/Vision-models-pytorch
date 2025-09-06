import boto3
from botocore.exceptions import ClientError


class AWSHandler:
    def __init__(self, s3_bucket_name: str):
        self.s3_bucket_name = s3_bucket_name

    def upload_file(self, local_path: str, to_path: str | None = None):
        s3 = boto3.client('s3')
        to_path = to_path or local_path.replace("\\", "/")
        
        s3.upload_file(local_path, self.s3_bucket_name, to_path)
        print(f"Uploaded {local_path} to s3://{self.s3_bucket_name}/{to_path}")

    def download_file(self, from_path: str, local_path: str | None = None):
        s3 = boto3.client('s3')
        local_path = local_path or from_path
        try:
            s3.download_file(self.s3_bucket_name, from_path, local_path)
        except ClientError as e:
            if e.response['Error']['Code'] == "404":
                print(f"The object s3://{self.s3_bucket_name}/{from_path} does not exist.")
            else:
                raise e
        print(f"Downloaded s3://{self.s3_bucket_name}/{from_path} to {local_path}")
