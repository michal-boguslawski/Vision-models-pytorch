from botocore.client import BaseClient
from botocore.exceptions import ClientError
import boto3
from boto3.dynamodb.conditions import Key
import time
from typing import Any
from utils.helpers import convert_floats_to_decimal


class AWSHandler:
    def __init__(
        self,
        s3_bucket_name: str | None = None,
        dynamodb_config_table: str | None = None,
        region: str = "eu-north-1",
    ):
        self.s3_bucket_name = s3_bucket_name
        self.dynamodb_config_table = dynamodb_config_table
        self.region = region

#########################S3##################################
    def create_s3_bucket_if_not_exists(self) -> dict[str, Any]:
        s3 = boto3.client('s3', region_name=self.region)
        try:
            return s3.head_bucket(Bucket=self.s3_bucket_name)
        except ClientError as e:
            if e.response['Error']['Code'] == "404":
                create_params = {
                    "Bucket": self.s3_bucket_name,
                    "CreateBucketConfiguration": {"LocationConstraint": self.region}
                }
                response = s3.create_bucket(**create_params)
                print(f"Created S3 bucket: {self.s3_bucket_name}")
                return response
            else:
                raise e

    def delete_s3_bucket(self) -> dict[str, Any]:
        s3 = boto3.client('s3', region_name=self.region)
        try:
            response = s3.delete_bucket(Bucket=self.s3_bucket_name)
            print(f"Deleted S3 bucket: {self.s3_bucket_name}")
            return response
        except ClientError as e:
            if e.response['Error']['Code'] == "404":
                print(f"The bucket {self.s3_bucket_name} does not exist.")
                return e.response
            else:
                raise e

    def upload_file_to_s3(self, local_path: str, s3_path: str | None = None) -> None:
        s3 = boto3.client('s3', region_name=self.region)
        s3_path = s3_path or local_path.replace("\\", "/")
        
        s3.upload_file(local_path, self.s3_bucket_name, s3_path)
        print(f"Uploaded {local_path} to s3://{self.s3_bucket_name}/{s3_path}")

    def download_file_from_s3(self, s3_path: str, local_path: str | None = None) -> None:
        s3 = boto3.client('s3', region_name=self.region)
        local_path = local_path or s3_path
        try:
            s3.download_file(self.s3_bucket_name, s3_path, local_path)
            print(f"Downloaded s3://{self.s3_bucket_name}/{s3_path} to {local_path}")

        except ClientError as e:
            if e.response['Error']['Code'] == "404":
                print(f"The object s3://{self.s3_bucket_name}/{s3_path} does not exist.")
                return e.response
            else:
                raise e

    def check_if_file_exists_in_s3(self, s3_path: str) -> bool:
        s3 = boto3.client('s3', region_name=self.region)
        try:
            s3.head_object(Bucket=self.s3_bucket_name, Key=s3_path)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == "404":
                return False
            else:
                raise e

    def delete_file_from_s3(self, s3_path: str) -> dict[str, Any]:
        s3 = boto3.client('s3', region_name=self.region)
        response = s3.delete_object(Bucket=self.s3_bucket_name, Key=s3_path)
        print(f"Deleted s3://{self.s3_bucket_name}/{s3_path}")
        return response

########################DynamoDB#############################

    def _wait_for_table_active(self, dynamodb_client: BaseClient, timeout=60, interval=2):
        """
        Wait until DynamoDB table is ACTIVE, or raise TimeoutError.
        
        :param dynamodb_client: boto3 DynamoDB client
        :param table_name: Name of the table to wait for
        :param timeout: Max seconds to wait
        :param interval: Seconds between retries
        """
        start_time = time.time()
        while True:
            response = dynamodb_client.describe_table(TableName=self.dynamodb_config_table)
            status = response["Table"]["TableStatus"]
            if status == "ACTIVE":
                return response
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Table {self.dynamodb_config_table} did not become ACTIVE within {timeout} seconds")
            time.sleep(interval)

    def create_dynamodb_table_if_not_exists(self) -> dict[str, Any]:
        dynamodb = boto3.client('dynamodb', region_name=self.region)
        try:
            response = dynamodb.describe_table(TableName=self.dynamodb_config_table)
            return response
        except dynamodb.exceptions.ResourceNotFoundException:
            response = dynamodb.create_table(
                TableName=self.dynamodb_config_table,
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
            response = self._wait_for_table_active(dynamodb)
            print(f"Created DynamoDB table: {self.dynamodb_config_table}")
            return response
        except Exception as e:
            raise e

    def put_item_to_dynamodb(self, item: dict[str, Any]) -> dict[str, Any]:
        dynamodb = boto3.resource('dynamodb', region_name=self.region)
        table = dynamodb.Table(self.dynamodb_config_table)  # type: ignore
        response = table.put_item(Item=convert_floats_to_decimal(item))
        print(f"Added item to DynamoDB table: {self.dynamodb_config_table}")
        return response

    def get_item_from_dynamodb(self, key: dict[str, Any]) -> dict[str, Any]:
        dynamodb = boto3.resource('dynamodb', region_name=self.region)
        table = dynamodb.Table(self.dynamodb_config_table)  # type: ignore
        response = table.get_item(TableName=self.dynamodb_config_table, Key=key)
        print(f"Retrieved item from DynamoDB table: {self.dynamodb_config_table}")
        return response

    def delete_item_from_dynamodb(self, key: dict[str, Any]) -> dict[str, Any]:
        dynamodb = boto3.resource('dynamodb', region_name=self.region)
        table = dynamodb.Table(self.dynamodb_config_table)  # type: ignore
        response = table.delete_item(TableName=self.dynamodb_config_table, Key=key)
        print(f"Deleted item from DynamoDB table: {self.dynamodb_config_table}")
        return response

    def delete_dynamodb_table(self) -> dict[str, Any]:
        dynamodb = boto3.client('dynamodb', region_name=self.region)
        response = dynamodb.delete_table(TableName=self.dynamodb_config_table)
        print(f"Deleted DynamoDB table: {self.dynamodb_config_table}")
        return response

    def query_dynamodb_table(self, project_name: str | None, experiment_name: str | None = None) -> dict[str, Any]:
        dynamodb = boto3.resource('dynamodb', region_name=self.region)
        key_condition_expression = None
        if project_name:
            key_condition_expression = Key("project_name").eq(project_name)
        table = dynamodb.Table(self.dynamodb_config_table)  # type: ignore
        response = table.query(KeyConditionExpression=key_condition_expression)
        print(f"Queried DynamoDB table: {self.dynamodb_config_table}")
        return response
