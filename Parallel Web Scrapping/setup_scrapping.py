import os
import boto3
import botocore

# configuration
REGION        = "us-east-1"
BUCKET_NAME   = "30123-books-scraping-bucket"
S3_PREFIX     = "scraping/"
DB_IDENTIFIER = "booksdb-instance"
DB_NAME       = "books"
DB_USERNAME   = "adminuser"
DB_PASSWORD   = "Wang011023"
SNS_TOPIC     = "scraping-complete-topic"
EMAIL_ADDR    = "zw2685@uchicago.edu"

# Initialize AWS clients
ec2 = boto3.client("ec2", region_name=REGION)
rds = boto3.client("rds", region_name=REGION)
s3  = boto3.client("s3", region_name=REGION)
sns = boto3.client("sns", region_name=REGION)

# Create or skip the S3 bucket
try:
    if REGION == "us-east-1":
        s3.create_bucket(Bucket=BUCKET_NAME)
    else:
        s3.create_bucket(
            Bucket=BUCKET_NAME,
            CreateBucketConfiguration={"LocationConstraint": REGION}
        )
    print(f"Bucket {BUCKET_NAME} OK")
except botocore.exceptions.ClientError as e:
    if e.response["Error"]["Code"] in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
        print(f"Bucket {BUCKET_NAME} already exists, skipping.")
    else:
        raise

# Create or reuse security group for MySQL
SG_NAME = "scraping-mysql-sg"
try:
    resp = ec2.create_security_group(GroupName=SG_NAME,
                                     Description="Allow MySQL from anywhere")
    sg_id = resp["GroupId"]
    ec2.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[{
            "IpProtocol": "tcp",
            "FromPort": 3306,
            "ToPort": 3306,
            "IpRanges": [{"CidrIp": "0.0.0.0/0"}]
        }]
    )
    print(f"Created SG {SG_NAME}: {sg_id}")
except botocore.exceptions.ClientError as e:
    if e.response["Error"]["Code"] == "InvalidGroup.Duplicate":
        sg_id = ec2.describe_security_groups(GroupNames=[SG_NAME])["SecurityGroups"][0]["GroupId"]
        print(f"SG {SG_NAME} exists: {sg_id}, skipping creation.")
    else:
        raise

# Create the MySQL RDS instance on t2.micro
try:
    rds.create_db_instance(
        DBInstanceIdentifier=DB_IDENTIFIER,
        AllocatedStorage=20,
        DBName=DB_NAME,
        DBInstanceClass="db.t2.micro",
        Engine="mysql",
        MasterUsername=DB_USERNAME,
        MasterUserPassword=DB_PASSWORD,
        VpcSecurityGroupIds=[sg_id],
        PubliclyAccessible=True
    )
    print("Creating RDS instance on t2.micro…")
except botocore.exceptions.ClientError as e:
    if e.response["Error"]["Code"] == "DBInstanceAlreadyExists":
        print("RDS instance exists; skipping creation.")
    else:
        raise

# wait for it to be available
waiter = rds.get_waiter("db_instance_available")
waiter.wait(DBInstanceIdentifier=DB_IDENTIFIER)
endpoint = rds.describe_db_instances(DBInstanceIdentifier=DB_IDENTIFIER)[
    "DBInstances"
][0]["Endpoint"]["Address"]
print("RDS endpoint:", endpoint)

# Upload scraper.py + batch JSONs to S3
for fname in ["scraper.py"] + [f for f in os.listdir() if f.startswith("batch") and f.endswith(".json")]:
    key = f"{S3_PREFIX}{fname}"
    s3.upload_file(Filename=fname, Bucket=BUCKET_NAME, Key=key)
    print(f"Uploaded {fname} to s3://{BUCKET_NAME}/{key}")

# Create SNS topic & subscribe my email
topic_arn = sns.create_topic(Name=SNS_TOPIC)["TopicArn"]
sns.subscribe(TopicArn=topic_arn, Protocol="email", Endpoint=EMAIL_ADDR)
print(f"SNS topic {topic_arn} created; subscription sent to {EMAIL_ADDR}")

print("Setup complete. Run:")
print(f"  python launch_scrapers.py --bucket {BUCKET_NAME} \\")
print(f"    --db_username {DB_USERNAME} --db_password {DB_PASSWORD} \\")
print(f"    --db_endpoint {endpoint} \\")
print(f"    --db_name {DB_NAME} --sns_topic_arn {topic_arn} \\")
print(f"    --region {REGION} \\")
print(f"    --instance_type t2.micro \\")
print(f"    --security_group_ids {sg_id} \\")
print(f"    --script_key {S3_PREFIX}scraper.py \\")
print(f"    --batch_prefix batch \\")
print(f"    --num_batches 8 \\")
print(f"    --iam_role LabInstanceProfile \\")
print(f"    --image_id ami-0062355a529d6089c")
