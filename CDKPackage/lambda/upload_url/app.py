import json, os, time, uuid, mimetypes
import boto3

s3 = boto3.client("s3")
BUCKET = os.environ["APP_BUCKET"]

def handler(event, context):
    body = json.loads(event.get("body") or "{}")
    filename = body.get("filename", f"file-{int(time.time())}")
    content_type = body.get("contentType") or mimetypes.guess_type(filename)[0] or "application/octet-stream"
    key = f"uploads/{uuid.uuid4()}{os.path.splitext(filename)[1]}"

    url = s3.generate_presigned_url(
        "put_object",
        Params={"Bucket": BUCKET, "Key": key, "ContentType": content_type},
        ExpiresIn=900
    )
    return {
        "statusCode": 200,
        "headers": {"content-type": "application/json"},
        "body": json.dumps({"url": url, "key": key, "bucket": BUCKET})
    }
