import argparse
import boto3
from botocore.exceptions import ClientError

def generate_user_data(region, bucket, script_key, batch_key, batch_filename,
                       db_username, db_password, db_endpoint, db_port, db_name,
                       sns_topic_arn):
    """Generate user data script for EC2 instance."""
    log_key = batch_filename.replace('.json', '.log')
    return f"""#!/bin/bash 
# Install dependencies
yum update -y && yum install -y python3 git
pip3 install requests bs4 mysql-connector-python boto3 'urllib3<1.27'

# Fetch scraper script and batch file
aws s3 cp s3://30123-books-scraping-bucket/scraping/scraper.py /home/ec2-user/scraper.py
aws s3 cp s3://30123-books-scraping-bucket/scraping/{batch_filename} /home/ec2-user/{batch_filename}

cd /home/ec2-user
python3 scraper.py \
  --input_file /home/ec2-user/{batch_filename} \
  --db_username {db_username} \
  --db_password {db_password} \
  --db_endpoint {db_endpoint} \
  --db_port {db_port} \
  --db_name {db_name} \
&& aws sns publish --region {region} --topic-arn {sns_topic_arn} \
      --message "Scraping succeeded: {batch_filename}"

# Notify via SNS
aws sns publish --region {region} --topic-arn {sns_topic_arn} --message "Scraping complete: {batch_filename}"

# Self-terminate the instance via AWS CLI
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region {region}
"""


def main():
    parser = argparse.ArgumentParser(description="Launch EC2 instances to run scraping batches")
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--script_key', required=True, help='S3 key for scraper.py')
    parser.add_argument('--num_batches', type=int, default=8, help='Number of batches/instances')
    parser.add_argument('--instance_type', default='t2.micro', help='EC2 instance type')
    parser.add_argument('--security_group_ids', nargs='+', required=True, help='EC2 security group IDs')
    parser.add_argument('--iam_role', required=True, help='Name of IAM role for EC2 instances')
    parser.add_argument('--db_username', required=True, help='Database username')
    parser.add_argument('--db_password', required=True, help='Database password')
    parser.add_argument('--db_endpoint', required=True, help='Database endpoint (hostname)')
    parser.add_argument('--db_port', default='3306', help='Database port')
    parser.add_argument('--batch_prefix', default='batch', help='Prefix for batch filenames')
    parser.add_argument('--db_name', required=True, help='Database name')
    parser.add_argument('--sns_topic_arn', required=True, help='SNS topic ARN for notification')
    parser.add_argument('--image_id', required=True, help='EC2 AMI ID')
    args = parser.parse_args()

    ec2 = boto3.client('ec2', region_name=args.region)

    for i in range(1, args.num_batches + 1):
        batch_filename = f"{args.batch_prefix}{i}.json"
        batch_key = batch_filename
        user_data = generate_user_data(
            args.region,
            args.bucket,
            args.script_key,
            batch_key,
            batch_filename,
            args.db_username,
            args.db_password,
            args.db_endpoint,
            args.db_port,
            args.db_name,
            args.sns_topic_arn
        )
        # Encode user data script
        try:
            response = ec2.run_instances(
                ImageId=args.image_id,
                InstanceType=args.instance_type,
                MinCount=1,
                MaxCount=1,
                SecurityGroupIds=args.security_group_ids,
                IamInstanceProfile={'Name': args.iam_role},
                UserData=user_data,
                TagSpecifications=[{
                    'ResourceType': 'instance',
                    'Tags': [
                        {'Key': 'Name', 'Value': f"scraper-batch-{i}"},
                        {'Key': 'Batch', 'Value': str(i)}
                    ]
                }]
            )
            print(f"Launched instance for batch {i}: {response['Instances'][0]['InstanceId']}")
        except ClientError as e:
            print(f"Error launching instance for batch {i}: {e}")

if __name__ == '__main__':
    main()
