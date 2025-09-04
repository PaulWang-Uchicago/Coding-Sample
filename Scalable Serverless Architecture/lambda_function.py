import json
import os
import boto3

s3 = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")
table_name = os.environ["DYNAMODB_TABLE"]
bucket = os.environ["BUCKET_NAME"]
table = dynamodb.Table(table_name)

def lambda_handler(event, context):
    for record in event.get("Records", []):
        try:
            survey = json.loads(record["body"])
            uid = survey.get("user_id")
            ts = survey.get("timestamp")
            elapsed = survey.get("time_elapsed", 0)
            text = survey.get("freetext", "").strip()

            # Skip invalid surveys
            if elapsed <= 3 or not text:
                continue

            # Upload to S3
            key = f"{uid}/{ts}.json"
            s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(survey),
                ContentType="application/json"
            )

            # Update DynamoDB (atomic + non-destructive)
            table.update_item(
                Key={"user_id": uid},
                UpdateExpression="""
                SET q1 = :q1,
                    q2 = :q2,
                    q3 = :q3,
                    q4 = :q4,
                    q5 = :q5,
                    #ts = :ts,
                    freetext = :ft,
                    time_elapsed = :te
                ADD submission_count :inc
                """,
                ExpressionAttributeNames={ "#ts": "timestamp" },
                ExpressionAttributeValues={
                    ":q1": survey.get("q1"),
                    ":q2": survey.get("q2"),
                    ":q3": survey.get("q3"),
                    ":q4": survey.get("q4"),
                    ":q5": survey.get("q5"),
                    ":ts": ts,
                    ":ft": text,
                    ":te": elapsed,
                    ":inc": 1
                }
            )

        except Exception as e:
            print(f"Error: {e}")
            raise  # Retry via SQS

    return {"statusCode": 200}