import json, os, time, uuid
import boto3

s3 = boto3.client("s3")
BUCKET = os.environ["APP_BUCKET"]

def handler(event, context):
    data = json.loads(event.get("body") or "{}")
    # se espera: key, prediction (opc), label
    doc = {
        "id": str(uuid.uuid4()),
        "key": data.get("key"),
        "prediction": data.get("prediction"),
        "label": data.get("label"),
        "user": data.get("user"),
        "ts": int(time.time())
    }
    s3.put_object(
        Bucket=BUCKET,
        Key=f"labels/{doc['id']}.json",
        Body=json.dumps(doc).encode("utf-8"),
        ContentType="application/json"
    )
    return {"statusCode": 200, "headers": {"content-type": "application/json"}, "body": json.dumps({"ok": True, "id": doc["id"]})}
