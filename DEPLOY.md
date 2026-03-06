# NirnAI API — AWS Deployment Guide

Production deployment on **AWS ECS Fargate** with an Application Load Balancer.

## Prerequisites

- AWS CLI installed and configured (`aws configure`)
- Docker installed locally
- An AWS account with permissions for ECR, ECS, VPC, ALB

## 1. Set variables

```bash
export AWS_REGION=ap-south-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ECR_REPO=nirnai-review-api
export ECS_CLUSTER=nirnai-cluster
export ECS_SERVICE=nirnai-api
export ECS_TASK=nirnai-api-task
```

## 2. Create ECR repository

```bash
aws ecr create-repository \
  --repository-name $ECR_REPO \
  --region $AWS_REGION
```

## 3. Build and push Docker image

```bash
# Authenticate Docker to ECR
aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS --password-stdin \
    $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Build image
docker build -t $ECR_REPO .

# Tag for ECR
docker tag $ECR_REPO:latest \
  $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest

# Push
docker push \
  $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest
```

## 4. Create ECS cluster

```bash
aws ecs create-cluster \
  --cluster-name $ECS_CLUSTER \
  --region $AWS_REGION
```

## 5. Create IAM execution role (one-time)

```bash
# Create the role
aws iam create-role \
  --role-name ecsTaskExecutionRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "ecs-tasks.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

# Attach required policy
aws iam attach-role-policy \
  --role-name ecsTaskExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
```

## 6. Register task definition

Create `ecs-task-def.json`:

```json
{
  "family": "nirnai-api-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::ACCOUNT_ID:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "nirnai-api",
      "image": "ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/nirnai-review-api:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "OPENAI_API_KEY", "value": "your-openai-key"},
        {"name": "PINECONE_API_KEY", "value": "your-pinecone-key"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/nirnai-api",
          "awslogs-region": "ap-south-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

Replace `ACCOUNT_ID` and `REGION`, then register:

```bash
aws ecs register-task-definition \
  --cli-input-json file://ecs-task-def.json
```

## 7. Create CloudWatch log group

```bash
aws logs create-log-group \
  --log-group-name /ecs/nirnai-api \
  --region $AWS_REGION
```

## 8. Create ALB + target group

Use the AWS Console or CLI to:

1. Create an Application Load Balancer in your VPC
2. Create a target group (target type = IP, port 8000, health check path = `/api/health`)
3. Add a listener (port 80 or 443) forwarding to the target group

Note the **target group ARN** and your **subnet IDs** and **security group ID**.

## 9. Create ECS service

```bash
aws ecs create-service \
  --cluster $ECS_CLUSTER \
  --service-name $ECS_SERVICE \
  --task-definition $ECS_TASK \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={
    subnets=[subnet-XXXXX,subnet-YYYYY],
    securityGroups=[sg-ZZZZZ],
    assignPublicIp=ENABLED
  }" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:...,containerName=nirnai-api,containerPort=8000"
```

## 10. Verify

```bash
# Get the ALB DNS name from the AWS Console, then:
curl https://your-alb-dns.amazonaws.com/api/health
```

Expected response:
```json
{"status": "healthy", "pinecone": "ok", "openai": "ok"}
```

## API Endpoints

| Method | Path          | Description                                |
|--------|---------------|--------------------------------------------|
| POST   | /api/review   | Run two-stage review on a merged case JSON |
| POST   | /api/ingest   | Ingest precedent cases into Pinecone       |
| GET    | /api/stats    | Pinecone index statistics                  |
| GET    | /api/health   | Health check (Pinecone + OpenAI)           |
| GET    | /docs         | Swagger UI (interactive API docs)          |

## Example: calling /api/review from your web app

```javascript
const response = await fetch('https://your-api-domain.com/api/review', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(mergedCaseJson),
});
const review = await response.json();
// review.overall_risk_level, review.sections, etc.
```

## Updating the service

When you push a new Docker image:

```bash
# Build, tag, push (same as step 3)
docker build -t $ECR_REPO .
docker tag $ECR_REPO:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest

# Force new deployment
aws ecs update-service \
  --cluster $ECS_CLUSTER \
  --service $ECS_SERVICE \
  --force-new-deployment
```

## Cost estimate

| Component      | Approximate cost              |
|----------------|-------------------------------|
| Fargate        | ~$30-50/month (1 vCPU, 2 GB) |
| ALB            | ~$16/month base               |
| OpenAI API     | ~$0.15-0.30 per review        |
| Pinecone       | Free tier (up to 100k vectors)|
| **Total base** | **~$50-70/month**             |
