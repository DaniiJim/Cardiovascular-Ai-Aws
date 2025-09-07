#!/usr/bin/env python3
import os
import aws_cdk as cdk
from infra.stack import MinimalMlWebappStack

app = cdk.App()

MinimalMlWebappStack(
    app,
    "MinimalMlWebappStack",
    # define cuenta/region por variables de entorno de CDK
    env=cdk.Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
        region=os.environ.get("CDK_DEFAULT_REGION")
    )
)

app.synth()
