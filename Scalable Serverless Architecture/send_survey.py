import json
import boto3
from botocore.exceptions import BotoCoreError, ClientError

def send_survey(survey_path, sqs_url):
    '''
    Input: survey_path (str): path to JSON survey data
    (e.g. ‘./survey.json’)
    sqs_url (str): URL for SQS queue
    Output: StatusCode (int): indicating whether the survey
    was successfully sent into the SQS queue (200) or
    not (0)
    '''
    # Load the survey JSON from device
    try:
        with open(survey_path, 'r') as f:
            survey_data = json.load(f)
    # If the file does not exist or is not valid JSON, return 0 to indicate failure
    except (IOError, json.JSONDecodeError):
        return 0

    # Send it into the SQS queue
    sqs = boto3.client('sqs')
    # Check if the SQS URL is valid
    try:
        response = sqs.send_message(
            QueueUrl=sqs_url,
            MessageBody=json.dumps(survey_data)
        )
        # Return the HTTP status code
        return response.get('ResponseMetadata', {}).get('HTTPStatusCode', 0)
    # If AWS itself throws an error, return 0
    except (BotoCoreError, ClientError):
        return 0