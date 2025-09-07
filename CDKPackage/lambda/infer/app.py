import json, os
import boto3
from botocore.exceptions import ClientError

s3 = boto3.client("s3")
rt = boto3.client("sagemaker-runtime")
BUCKET = os.environ["APP_BUCKET"]
ENDPOINT = os.environ["ENDPOINT_NAME"]

# Ajusta content-type según tu modelo (text/csv, application/json, etc.)
CONTENT_TYPE = os.environ.get("INFER_CONTENT_TYPE", "application/octet-stream")

def handler(event, context):
    try:
        body = json.loads(event.get("body") or "{}")
        key = body["key"]  # p.ej. uploads/xxx
        obj = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()

        resp = rt.invoke_endpoint(
            EndpointName=ENDPOINT,
            ContentType=CONTENT_TYPE,
            Body=obj
        )
        payload = resp["Body"].read()
        # Intenta parsear JSON; si no, devuélvelo como texto
        try:
            result = json.loads(payload)
        except Exception:
            result = {"raw": payload.decode("utf-8", errors="ignore")}

        return {"statusCode": 200, "headers": {"content-type": "application/json"}, "body": json.dumps(result)}
    except ClientError as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
