import json, os
import boto3

AGENT = boto3.client("bedrock-agent-runtime", region_name=os.environ.get("AWS_REGION"))
BR = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION"))
KB_ID = os.environ["BEDROCK_KB_ID"]
MODEL_ID = os.environ["BEDROCK_MODEL_ID"]

def _retrieve_context(query: str, k: int = 5) -> str:
    res = AGENT.retrieve(
        knowledgeBaseId=KB_ID,
        retrievalQuery={"text": query},
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": k}},
    )
    parts = []
    for r in res.get("retrievalResults", []):
        # distintas formas posibles en la respuesta; sacamos texto si existe
        c = r.get("content", {})
        if isinstance(c, dict) and "text" in c:
            parts.append(c["text"])
        elif isinstance(c, list):
            parts.extend([x.get("text","") for x in c if isinstance(x, dict) and x.get("text")])
    return "\n\n".join([p for p in parts if p])

def handler(event, context):
    p = json.loads(event.get("body") or "{}")
    user_msg = p.get("message", "")
    prediction = p.get("prediction", "")
    label = p.get("label", "")
    query = f"{user_msg}\nPrediccion del modelo: {prediction}\nEtiqueta del usuario: {label}"
    ctx = _retrieve_context(query, k=5)

    prompt = f"""Analiza el resultado del modelo y contesta de forma clara.
Contexto útil (puede estar incompleto):
{ctx}

Resultado del modelo: {prediction}
Etiqueta del usuario (si existe): {label}

Respuesta:"""

    # Ejemplo para modelos AnthropIc "Claude 3" en Bedrock
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
    }

    out = BR.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body).encode("utf-8")
    )
    payload = json.loads(out["body"].read())
    # Extraer el texto (formato Claude)
    try:
        answer = "".join([b.get("text","") for b in payload["content"] if b.get("type")=="text"])
    except Exception:
        answer = json.dumps(payload)

    return {
        "statusCode": 200,
        "headers": {"content-type": "application/json"},
        "body": json.dumps({"answer": answer})
    }
