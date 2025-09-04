import boto3
import botocore

def setup_resources():
    # Initialize clients
    sqs = boto3.client("sqs")
    s3 = boto3.client("s3")
    dynamodb = boto3.client("dynamodb")
    lambda_client = boto3.client("lambda")

    # Create SQS Queue
    queue = sqs.create_queue(QueueName="survey-queue")
    queue_url = queue["QueueUrl"]
    queue_arn = sqs.get_queue_attributes(
        QueueUrl=queue_url,
        AttributeNames=["QueueArn"]
    )["Attributes"]["QueueArn"]
    print("Created SQS Queue:", queue_url)

    # Create S3 Bucket
    bucket_name = "paulwang-survey-bucket-30123"
    try:
        s3.create_bucket(Bucket=bucket_name)
    except s3.exceptions.BucketAlreadyExists:
        print(f"Bucket {bucket_name} already exists. Skipping creation.")

    # Create DynamoDB Table (if not exists)
    table_name = "survey-data"
    existing_tables = dynamodb.list_tables()["TableNames"]
    if table_name not in existing_tables:
        dynamodb.create_table(
            TableName=table_name,
            KeySchema=[{"AttributeName": "user_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "user_id", "AttributeType": "S"}],
            ProvisionedThroughput={"ReadCapacityUnits": 5, "WriteCapacityUnits": 5}
        )
        print(f"Created DynamoDB table: {table_name}")
    else:
        print(f"Table {table_name} already exists. Skipping creation.")

# Deploy Lambda Function or skip if it already exists
    role_arn = "arn:aws:iam::471112944418:role/LabRole"
    with open("lambda_function.zip", "rb") as f:
        lambda_code = f.read()

    try:
        lambda_client.create_function(
            FunctionName="process-survey",
            Runtime="python3.9",
            Role=role_arn,
            Handler="lambda_function.lambda_handler",
            Code={"ZipFile": lambda_code},
            Environment={
                "Variables": {
                    "DYNAMODB_TABLE": table_name,
                    "BUCKET_NAME": bucket_name
                }
            }
        )
        print("Created Lambda function process-survey")
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "ResourceConflictException":
            # Function already exists—just skip
            print("Lambda function process-survey already exists; skipping creation.")
        else:
            # Any other error should be raised
            raise

    # link SQS to Lambda  
    mappings = lambda_client.list_event_source_mappings(
        EventSourceArn=queue_arn,
        FunctionName="process-survey"
    )["EventSourceMappings"]
    if not mappings:
        print("Linking SQS queue as trigger…")
        lambda_client.create_event_source_mapping(
            EventSourceArn=queue_arn,
            FunctionName="process-survey",
            Enabled=True,
            BatchSize=10
        )
    else:
        print("SQS trigger already in place.")

if __name__ == "__main__":
    setup_resources()