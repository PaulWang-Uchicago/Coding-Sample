import boto3

def configure_trigger(queue_arn, lambda_function_arn):
    """Configure SQS to trigger Lambda."""
    lambda_client = boto3.client('lambda')
    
    # Grant SQS permission to invoke Lambda
    lambda_client.add_permission(
        FunctionName=lambda_function_arn,
        StatementId='SQSTriggerLambda',
        Action='lambda:InvokeFunction',
        Principal='sqs.amazonaws.com',
        SourceArn=queue_arn
    )
    
    # Create event source mapping (connect SQS to Lambda)
    response = lambda_client.create_event_source_mapping(
        EventSourceArn=queue_arn,
        FunctionName=lambda_function_arn,
        Enabled=True,
        BatchSize=10,
        MaximumRetryAttempts=2
    )
    return response