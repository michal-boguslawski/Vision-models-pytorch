import boto3


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
        from_path = local_path or from_path
        
        s3.download_file(self.s3_bucket_name, from_path, local_path)
        print(f"Downloaded s3://{self.s3_bucket_name}/{from_path} to {local_path}")
