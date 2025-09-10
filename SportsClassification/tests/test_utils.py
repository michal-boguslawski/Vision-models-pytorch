from botocore.exceptions import ClientError
import pytest
import os
from .fixtures import test_image_file
from utils.aws_handler import AWSHandler
from utils.helpers import generate_random_name
from utils.filesystem import make_dirs


@pytest.fixture(scope="module")
def aws_handler():
    s3_bucket_name = generate_random_name(12)
    dynamodb_config_table = generate_random_name(12)
    handler = AWSHandler(s3_bucket_name=s3_bucket_name, dynamodb_config_table=dynamodb_config_table)
    return handler


class TestsAWSHandler:
    @pytest.mark.order(1)
    def test_create_aws_handler(self, aws_handler):
        assert aws_handler is not None

#######################S3###############################
    @pytest.mark.order(2)
    def test_create_new_s3_bucket(self, aws_handler):

        create_response = aws_handler.create_s3_bucket_if_not_exists()
        assert create_response["ResponseMetadata"]["HTTPStatusCode"] == 200
        assert create_response["Location"] == f"http://{aws_handler.s3_bucket_name}.s3.amazonaws.com/"

    @pytest.mark.order(3)
    def test_create_existing_s3_bucket(self, aws_handler):
        response = aws_handler.create_s3_bucket_if_not_exists()
        assert response["ResponseMetadata"]["HTTPStatusCode"] == 200
        assert "Location" not in response

    def test_create_s3_bucket_with_invalid_name(self):
        s3_bucket_name = "XXXX"
        aws_handler = AWSHandler(s3_bucket_name=s3_bucket_name)

        with pytest.raises(ClientError) as exc_info:
            aws_handler.create_s3_bucket_if_not_exists()

        # Optional: check specific error code
        error_code = exc_info.value.response["Error"]["Code"]
        assert error_code in ("InvalidBucketName", "BucketAlreadyExists", "BucketAlreadyOwnedByYou")

    @pytest.mark.order(4)
    def test_upload_file_to_s3(self, aws_handler, test_image_file):
        s3_path = "data/test_image.jpg"
        aws_handler.upload_file_to_s3(local_path=test_image_file, s3_path=s3_path)
        assert aws_handler.check_if_file_exists_in_s3(s3_path=s3_path)

    @pytest.mark.order(5)
    def test_download_file_from_s3(self, aws_handler, tmp_path):
        local_path = tmp_path / "test_image.jpg"
        s3_path = "data/test_image.jpg"
        aws_handler.download_file_from_s3(local_path=str(local_path), s3_path=s3_path)
        assert local_path.exists()
        assert local_path.stat().st_size > 0

    @pytest.mark.order(6)
    def test_delete_file_from_s3(self, aws_handler):
        s3_path = "data/test_image.jpg"
        response = aws_handler.delete_file_from_s3(s3_path=s3_path)
        assert response["ResponseMetadata"]["HTTPStatusCode"] == 204

    @pytest.mark.order(98)
    def test_delete_s3_bucket(self, aws_handler):
        response = aws_handler.delete_s3_bucket()
        assert response["ResponseMetadata"]["HTTPStatusCode"] == 204

############################DynamoDB###############################
    @pytest.mark.order(7)
    def test_create_dynamodb_table(self, aws_handler):
        response = aws_handler.create_dynamodb_table_if_not_exists()
        assert response["Table"]["TableStatus"] == "ACTIVE"

    @pytest.mark.order(8)
    def test_put_item_into_dynamodb(self, aws_handler):
        item = {"project_name": "test_project", "experiment_name": "test_experiment", "config": {"test": "test"}, "id": "1", "age": 30}
        response = aws_handler.put_item_to_dynamodb(item)
        assert response["ResponseMetadata"]["HTTPStatusCode"] == 200

    @pytest.mark.order(9)
    def test_get_item_from_dynamodb(self, aws_handler):
        item = {"project_name": "test_project", "experiment_name": "test_experiment"}
        response = aws_handler.get_item_from_dynamodb(item)
        assert response["Item"]["project_name"] == "test_project"

    @pytest.mark.order(11)
    def test_delete_item_from_dynamodb(self, aws_handler):
        item = {"project_name": "test_project", "experiment_name": "test_experiment"}
        response = aws_handler.delete_item_from_dynamodb(item)
        assert response["ResponseMetadata"]["HTTPStatusCode"] == 200

    @pytest.mark.order(99)
    def test_delete_dynamodb_table(self, aws_handler):
        response = aws_handler.delete_dynamodb_table()
        assert response["TableDescription"]["TableStatus"] == "DELETING"
