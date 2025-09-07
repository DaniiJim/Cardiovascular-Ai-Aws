from aws_cdk import (
    Stack,
    Duration,
    RemovalPolicy,
    CfnParameter,
    CfnOutput,
    aws_s3 as s3,
    aws_lambda as _lambda,
    aws_iam as iam,
    aws_apigateway as apigw,
)
from constructs import Construct
import os


class MinimalMlWebappStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        # ========= Parámetros =========
        endpoint_name = CfnParameter(
            self, "SageMakerEndpointName",
            type="String",
            description="Nombre del SageMaker Endpoint ya desplegado (p.ej. my-xgb-endpoint)",
        )

        kb_id = CfnParameter(
            self, "BedrockKnowledgeBaseId",
            type="String",
            description="ID de la Knowledge Base de Bedrock (p.ej. KB123ABC...)"
        )

        model_id = CfnParameter(
            self, "BedrockModelId",
            type="String",
            default="anthropic.claude-3-haiku-20240307-v1:0",
            description="ID del modelo Bedrock a invocar"
        )

        # ========= Buckets =========
        # Frontend (estático) - público solo lectura
        site_bucket = s3.Bucket(
            self, "SiteBucket",
            website_index_document="index.html",
            public_read_access=True,
            block_public_access=s3.BlockPublicAccess(
                block_public_acls=False,
                block_public_policy=False,
                ignore_public_acls=False,
                restrict_public_buckets=False
            ),
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        # App/data (privado)
        app_bucket = s3.Bucket(
            self, "AppBucket",
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            cors=[s3.CorsRule(
                allowed_methods=[s3.HttpMethods.GET, s3.HttpMethods.PUT, s3.HttpMethods.POST],
                allowed_origins=["*"],   # para MVP; afina con tu dominio/apigw
                allowed_headers=["*"],
                exposed_headers=["ETag"]
            )]
        )

        # ========= Roles/Lambdas =========
        common_env = {
            "APP_BUCKET": app_bucket.bucket_name,
            "ENDPOINT_NAME": endpoint_name.value_as_string,
            "BEDROCK_KB_ID": kb_id.value_as_string,
            "BEDROCK_MODEL_ID": model_id.value_as_string,
            "AWS_REGION": self.region
        }

        # --- Lambda: upload-url ---
        upload_fn = _lambda.Function(
            self, "UploadUrlFn",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="app.handler",
            code=_lambda.Code.from_asset("lambda/upload_url"),
            timeout=Duration.seconds(10),
            environment=common_env
        )
        app_bucket.grant_put(upload_fn)       # para PUT vía presigned no es estrictamente necesario, pero útil si validas server-side
        app_bucket.grant_read(upload_fn)

        # --- Lambda: infer ---
        infer_fn = _lambda.Function(
            self, "InferFn",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="app.handler",
            code=_lambda.Code.from_asset("lambda/infer"),
            timeout=Duration.seconds(30),
            environment=common_env,
            memory_size=512,
        )
        app_bucket.grant_read(infer_fn)
        # Permiso a SageMaker endpoint específico
        infer_fn.add_to_role_policy(iam.PolicyStatement(
            actions=["sagemaker:InvokeEndpoint"],
            resources=[f"arn:aws:sagemaker:{self.region}:{self.account}:endpoint/{endpoint_name.value_as_string}"]
        ))

        # --- Lambda: label ---
        label_fn = _lambda.Function(
            self, "LabelFn",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="app.handler",
            code=_lambda.Code.from_asset("lambda/label"),
            timeout=Duration.seconds(10),
            environment=common_env
        )
        app_bucket.grant_read_write(label_fn)

        # --- Lambda: chat (Bedrock RAG) ---
        chat_fn = _lambda.Function(
            self, "ChatFn",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="app.handler",
            code=_lambda.Code.from_asset("lambda/chat"),
            timeout=Duration.seconds(60),
            environment=common_env,
            memory_size=1024,
        )
        # Permisos Bedrock (Retrieve + InvokeModel)
        chat_fn.add_to_role_policy(iam.PolicyStatement(
            actions=["bedrock:Retrieve"],
            resources=[f"arn:aws:bedrock:{self.region}:{self.account}:knowledge-base/{kb_id.value_as_string}"]
        ))
        chat_fn.add_to_role_policy(iam.PolicyStatement(
            actions=["bedrock:InvokeModel"],
            resources=[f"arn:aws:bedrock:{self.region}::foundation-model/{model_id.value_as_string}"]
        ))
        app_bucket.grant_read(chat_fn)

        # ========= API REST =========
        api = apigw.RestApi(
            self, "MinimalApi",
            deploy_options=apigw.StageOptions(
                stage_name="prod",
                throttling_rate_limit=100,
                throttling_burst_limit=50,
                metrics_enabled=True,
                logging_level=apigw.MethodLoggingLevel.INFO,
                data_trace_enabled=False,
            ),
            default_cors_preflight_options=apigw.CorsOptions(
                allow_origins=apigw.Cors.ALL_ORIGINS,
                allow_methods=apigw.Cors.ALL_METHODS,
                allow_headers=["*"]
            ),
        )

        def add_post(path: str, fn: _lambda.Function):
            res = api.root.add_resource(path)
            res.add_method("POST", apigw.LambdaIntegration(fn))

        add_post("upload-url", upload_fn)
        add_post("infer", infer_fn)
        add_post("label", label_fn)
        add_post("chat", chat_fn)

        # ========= Outputs =========
        CfnOutput(self, "SiteBucketName", value=site_bucket.bucket_name)
        CfnOutput(self, "AppBucketName", value=app_bucket.bucket_name)
        CfnOutput(self, "ApiUrl", value=api.url)
