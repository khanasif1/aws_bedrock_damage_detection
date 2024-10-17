import streamlit as st
import boto3
import io
import base64
from PIL import Image
import botocore.config
import json
# Function to call Amazon Bedrock Claude 3 Sonnet
def call_bedrock_claude3_sonnet(prompt, image_bytes):
    # Set up AWS Bedrock client (Assuming you have AWS credentials configured)
    bedrock_client = boto3.client('bedrock-runtime', region_name='ap-southeast-2')
    
    # You can specify the model ARN for Claude 3 Sonnet
    # model_id = 'arn:aws:bedrock:ap-southeast-2:654654521070:foundation-model/claude3-sonnet'
    model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'

    # Convert image bytes to base64
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')

    # # Prepare the payload
    # payload = {
    #     "prompt": prompt,
    #     "image": image_base64,
    #     "max_tokens": 2048
    # }
    # # Convert payload to JSON string
    # payload_json = json.dumps(payload)

    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_base64,
                        },
                    },
                ],
            }
        ],
    }

    # Call the Bedrock inference model
    response = bedrock_client.invoke_model(
        # modelId=model_arn,
        modelId=model_id,
        contentType='application/json',
        accept='application/json',
        # body=payload_json
        body=json.dumps(request_body)
    )

    # Get the response (adjust this depending on Bedrock API response format)
    result = response['body'].read().decode('utf-8')
    
    return result

# Streamlit App interface
st.title('Amazon Bedrock Claude 3 Sonnet Image Inference')

# File uploader for the image
uploaded_image = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

# Text input for prompt
prompt = st.text_input("Enter your prompt:")

# If both an image and prompt are provided
if uploaded_image and prompt:
    # Load the image
    image = Image.open(uploaded_image)
    
    # Display the image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Convert image to bytes
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes = image_bytes.getvalue()

    # Button to trigger inference
    if st.button('Run Inference'):
        st.write('Running inference...')
        try:
            result = call_bedrock_claude3_sonnet(prompt, image_bytes)
            st.success('Inference completed!')
            st.write('Result:', result)
        except Exception as e:
            st.error(f'Error during inference: {e}')
