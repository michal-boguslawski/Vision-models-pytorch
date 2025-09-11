from botocore.exceptions import ClientError
import boto3
from boto3.dynamodb.conditions import Key
import logging
from mypy_boto3_s3.type_defs import BucketLocationConstraintType  # type: ignore
import os
from typing import Any, cast
from utils.helpers import convert_floats_to_decimal
from utils.metaclass import SingletonMeta


logger = logging.getLogger("AppLogger")


class AWSHandler(metaclass=SingletonMeta):
    def __init__(self):
        self.s3_bucket_name = os.getenv("S3_BUCKET")
        self.dynamodb_config_table_name = os.getenv("DYNAMODB_CONFIG_TABLE")
        self.region: BucketLocationConstraintType = cast(BucketLocationConstraintType, os.getenv("AWS_REGION", "eu-north-1"))
        
        self.s3_resource = boto3.resource('s3', region_name=self.region)
        self.dynamodb_resource = boto3.resource('dynamodb', region_name=self.region)
        
        self.s3_bucket = None
        self.dynamodb_config_table = None
        
        self._initialized = False

    def initialize(self):
        self.create_s3_bucket_if_not_exists()
        self.create_dynamodb_config_table_if_not_exists()
        self._initialized = True

#########################S3##################################
    def _bucket_exists(self) -> bool:
        if not self.s3_bucket_name:
            logger.error("S3 bucket name is not set.")
            return False
        try:
            self.s3_resource.meta.client.head_bucket(Bucket=self.s3_bucket_name)
            logger.info(f"S3 bucket {self.s3_bucket_name} exists.")
            return True
        except ClientError as e:
            error_code = int(e.response.get("Error", {}).get("Code", 0))
            if error_code == 404:
                return False
            raise e

    def create_s3_bucket_if_not_exists(self) -> None:
        if self._bucket_exists() and self.s3_bucket_name:
            self.s3_bucket = self.s3_resource.Bucket(self.s3_bucket_name)
            return
        elif self.s3_bucket_name:
            self.s3_bucket = self.s3_resource.create_bucket(
                Bucket=self.s3_bucket_name,
                CreateBucketConfiguration={
                    'LocationConstraint': self.region
                }
            )
            logger.info(f"Created S3 bucket: {self.s3_bucket_name}")

    def delete_s3_bucket(self) -> bool:
        if self._bucket_exists() and self.s3_bucket_name:
            self.s3_bucket = self.s3_resource.Bucket(self.s3_bucket_name)
            self.s3_bucket.objects.all().delete()
            self.s3_bucket.delete()
            self.s3_bucket = None
            logger.info(f"Deleted S3 bucket: {self.s3_bucket_name}")
            return True
        logger.error("S3 bucket is not set.")
        return False

    def upload_file_to_s3(self, local_path: str, s3_path: str | None = None) -> bool:
        if self.s3_bucket:
            self.s3_bucket.upload_file(local_path, s3_path or local_path)
            logger.info(f"Uploaded {local_path} to s3://{self.s3_bucket_name}/{s3_path or local_path}")
            return True
        logger.error("S3 bucket is not set.")
        return False

    def download_file_from_s3(self, s3_path: str, local_path: str | None = None) -> bool:
        if self.s3_bucket:
            self.s3_bucket.download_file(s3_path, local_path or s3_path)
            logger.info(f"Downloaded s3://{self.s3_bucket_name}/{s3_path} to {local_path or s3_path}")
            return True
        logger.error("S3 bucket is not set.")
        return False

    def check_if_object_exists_in_s3(self, s3_path: str) -> bool:
        if self.s3_bucket:
            try:
                self.s3_bucket.Object(s3_path).load()
            except ClientError as e:
                if int(e.response.get("Error", {}).get("Code", 0)) == 404:
                    return False
                raise e
            return True
        return False

    def delete_file_from_s3(self, s3_path: str) -> bool:
        if self.s3_bucket:
            self.s3_bucket.Object(s3_path).delete()
            logger.info(f"Deleted s3://{self.s3_bucket_name}/{s3_path}")
            return True
        return False

########################DynamoDB#############################
    def _dynamodb_table_exists(self) -> bool:
        if not self.dynamodb_config_table_name:
            logger.error("DynamoDB table name is not set.")
            return False
        try:
            self.dynamodb_resource.meta.client.describe_table(TableName=self.dynamodb_config_table_name)
            logger.info(f"DynamoDB table {self.dynamodb_config_table_name} exists.")
            return True
        except ClientError as e:
            error_code = int(e.response.get("Error", {}).get("Code", 0))
            if error_code == 400:
                return False
            raise e

    def create_dynamodb_config_table_if_not_exists(self) -> None:
        if self._dynamodb_table_exists() and self.dynamodb_config_table_name:
            self.dynamodb_config_table = self.dynamodb_resource.Table(self.dynamodb_config_table_name)
        elif self.dynamodb_config_table_name:
            self.dynamodb_config_table = self.dynamodb_resource.create_table(
                TableName=self.dynamodb_config_table_name,
                KeySchema=[
                    {"AttributeName": "project_name", "KeyType": "HASH"},  # Partition key
                    {"AttributeName": "experiment_name", "KeyType": "RANGE"}  # Sort key
                ],
                AttributeDefinitions=[
                    {"AttributeName": "project_name", "AttributeType": "S"},
                    {"AttributeName": "experiment_name", "AttributeType": "S"}
                ],
                BillingMode="PAY_PER_REQUEST"  # On-demand capacity mode
            )
            self.dynamodb_config_table.wait_until_exists()
            logger.info(f"Created DynamoDB table: {self.dynamodb_config_table}")

    def delete_dynamodb_table(self) -> bool:
        if self._dynamodb_table_exists() and self.dynamodb_config_table_name:
            self.dynamodb_config_table = self.dynamodb_resource.Table(self.dynamodb_config_table_name)
            self.dynamodb_config_table.delete()
            self.dynamodb_config_table = None
            logger.info(f"Deleted DynamoDB table: {self.dynamodb_config_table_name}")
            return True
        logger.error("DynamoDB table name is not set.")
        return False

    def put_item_to_dynamodb(self, item: dict[str, Any]) -> bool:
        if self.dynamodb_config_table:
            self.dynamodb_config_table.put_item(Item=convert_floats_to_decimal(item))
            logger.info(f"Added item to DynamoDB table: {self.dynamodb_config_table}")
            return True
        logger.error("DynamoDB table is not set.")
        return False

    def get_item_from_dynamodb(self, key: dict[str, Any]) -> dict[str, Any] | None:
        if self.dynamodb_config_table:
            return self.dynamodb_config_table.get_item(Key=key).get("Item")
        logger.error("DynamoDB table is not set.")

    def delete_item_from_dynamodb(self, key: dict[str, Any]) -> bool:
        if self.dynamodb_config_table:
            self.dynamodb_config_table.delete_item(Key=key)
            logger.info(f"Deleted item from DynamoDB table: {self.dynamodb_config_table}")
            return True
        logger.error("DynamoDB table is not set.")
        return False

    def query_dynamodb_table(self, project_name: str | None, experiment_name: str | None = None):
        pass
