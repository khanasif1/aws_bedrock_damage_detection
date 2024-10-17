import boto3

def list_bedrock_models():
    bedrock_client = boto3.client('bedrock', region_name='us-east-1')
    
    # List available models in your region
    response = bedrock_client.list_foundation_models()
    models = response.get('models', [])
    print('model data')
    for model in models:
        print(f"Model ARN: {model['modelArn']}, Model Name: {model['modelId']}")

# Call the function
list_bedrock_models()
