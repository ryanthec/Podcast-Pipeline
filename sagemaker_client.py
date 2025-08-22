import os
import json
import boto3
import logging
import time
import uuid
import wave
import httpx  # For async behavior
from datetime import datetime
from urllib.parse import urlparse
from dotenv import load_dotenv
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials
from botocore.config import Config
from botocore.exceptions import ClientError

load_dotenv()

class SageMakerSession:
    def __init__(self, endpoint_name: str = None, max_new_tokens=1800):

        if(endpoint_name == None):
            self.endpoint_name = os.environ['QWEN_ENDPOINT']
        else:
            self.endpoint_name = endpoint_name

        self.max_new_tokens = max_new_tokens        
        self.temperature = 0.4
        self.top_p = 0.5
        self.presence_penalty = 0.0
        self.region_name = "ap-southeast-1"

        config = Config(
            connect_timeout=300,
            read_timeout=300,
            retries={'max_attempts': 0}
        )

        # Initialize the SageMaker runtime client
        self.session = boto3.Session(
            aws_access_key_id=os.environ['AWS_SAGEMAKER_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['AWS_SAGEMAKER_SECRET_ACCESS_KEY'],
            region_name=self.region_name
        )
        self.runtime_client = self.session.client('sagemaker-runtime', config=config)


    def invoke(self, messages: list):
        """Sends messages to the SageMaker endpoint and returns the response."""
        payload = {
            "messages": messages, 
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "return_full_text": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        try:
            response = self.runtime_client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="application/json",
                Body=json.dumps(payload)
            )
            response_dict = json.loads(response['Body'].read().decode("utf-8"))
            response_content = response_dict['choices'][0]['message']['content']

            print("Full API Response:", response_dict)
            return response_content
            
        except Exception as e:
            print(f"An error occurred: {e}")
            if hasattr(e, 'response'):
                print(f"Error response: {e.response}")
            return None
        
    async def async_invoke(self, messages: list):
        """Asynchronous invocation to SageMaker (using httpx)."""
        url = f"https://runtime.sagemaker.{self.region_name}.amazonaws.com/endpoints/{self.endpoint_name}/invocations"
        payload = {
            "messages": messages,
            "parameters": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "presence_penalty": self.presence_penalty,
                "return_full_text": False
            },
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        }

        # Prepare AWS credentials
        credentials = Credentials(
            access_key=os.environ['AWS_SAGEMAKER_ACCESS_KEY_ID'],
            secret_key=os.environ['AWS_SAGEMAKER_SECRET_ACCESS_KEY']
        )

        aws_request = AWSRequest(
            method="POST",
            url=url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"}
        )
        SigV4Auth(credentials, "sagemaker", self.region_name).add_auth(aws_request)

        signed_headers = dict(aws_request.headers)

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(url, headers=signed_headers, json=payload)
                if response.is_error:
                    print(f"Error: {response.status_code} - {response.text}")
                    return None
                
                parsed_response = response.json()
                print("Full API Response:", parsed_response)
                content = self.extract_content(parsed_response)
                return content
                
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def invoke_embedding(self, text: str, embedding_endpoint: str):
        """Sends text to the embedding SageMaker endpoint and returns the embedding vector."""
        payload = {"inputs": text}

        try:
            response = self.runtime_client.invoke_endpoint(
                EndpointName=embedding_endpoint,
                ContentType="application/json",
                Body=json.dumps(payload)
            )
            response_body = json.loads(response['Body'].read().decode('utf-8'))
            return response_body[0]  # Return the first embedding vector
        except Exception as e:
            logging.error(f"Error querying SageMaker embedding endpoint: {e}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            return None


class SageMakerClient:
    def __init__(self, endpoint_name: str = None, max_new_tokens=1800):
        self.client = SageMakerSession(endpoint_name, max_new_tokens)
        self.system_generation_prompt = ('''
        You are a script generator for a podcast.
        You will be given a chunk of text from a document and you will need to generate a script for the podcast between 2 speakers based on that text.
        ''')

    def prompt(self, user_prompt: str, system_prompt: str = None):
        """Invokes the SageMaker endpoint and returns the generated text."""
        if system_prompt is None:
            system_prompt = self.system_generation_prompt
            
        # Format messages for Qwen3 chat format
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user", 
                "content": user_prompt
            }
        ]
        
        # Send the messages to the invoke method
        response = self.client.invoke(messages)
        return response

    async def async_prompt(self, user_prompt: str, system_prompt: str = None):
        """Asynchronously handles invocation to SageMaker."""
        if system_prompt is None:
            system_prompt = self.system_generation_prompt
            
        # Format messages for Qwen3 chat format
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        
        # Send the messages to the async invoke method
        response = await self.client.async_invoke(messages)
        return response


class SageMakerEmbeddingClient:
    def __init__(self, embedding_endpoint: str = None):
        self.embedding_endpoint = embedding_endpoint or os.getenv('EMBEDDING_ENDPOINT')
        self.region_name = "ap-southeast-1"
        
        config = Config(
            connect_timeout=300,
            read_timeout=300,
            retries={'max_attempts': 0}
        )
        
        # Initialize the SageMaker runtime client
        self.session = boto3.Session(
            aws_access_key_id=os.environ['AWS_SAGEMAKER_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['AWS_SAGEMAKER_SECRET_ACCESS_KEY'],
            region_name=self.region_name
        )
        self.runtime_client = self.session.client('sagemaker-runtime', config=config)

    def invoke(self, text: str):
        """Sends text to the embedding SageMaker endpoint and returns the embedding vector."""
        payload = {"inputs": text}

        try:
            response = self.runtime_client.invoke_endpoint(
                EndpointName=self.embedding_endpoint,
                ContentType="application/json",
                Body=json.dumps(payload)
            )
            response_body = json.loads(response['Body'].read().decode('utf-8'))
            return response_body[0]  # Return the first embedding vector
        except Exception as e:
            logging.error(f"Error querying SageMaker embedding endpoint: {e}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            return None

    async def async_invoke(self, text: str):
        """Asynchronous embedding invocation to SageMaker."""
        url = f"https://runtime.sagemaker.{self.region_name}.amazonaws.com/endpoints/{self.embedding_endpoint}/invocations"
        payload = {"inputs": text}

        credentials = Credentials(
            access_key=os.environ['AWS_SAGEMAKER_ACCESS_KEY_ID'],
            secret_key=os.environ['AWS_SAGEMAKER_SECRET_ACCESS_KEY']
        )

        aws_request = AWSRequest(
            method="POST",
            url=url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"}
        )
        SigV4Auth(credentials, "sagemaker", self.region_name).add_auth(aws_request)
        signed_headers = dict(aws_request.headers)

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(url, headers=signed_headers, json=payload)
                if response.is_error:
                    logging.error(f"Error: {response.status_code} - {response.text}")
                    return None
                response_body = response.json()
                return response_body[0]
        except Exception as e:
            logging.error(f"Error querying SageMaker embedding endpoint: {e}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            return None



# class SageMakerTTSClient:
#     def __init__(self, endpoint_name: str = None, s3_bucket: str = None):
#         if endpoint_name is None:
#             self.endpoint_name = os.environ['TTS_ENDPOINT']
#         else:
#             self.endpoint_name = endpoint_name

#         self.region_name = "ap-southeast-1"
        
#         # Set S3 bucket for inputs/outputs
#         if s3_bucket is None:
#             self.s3_bucket = os.environ.get('TTS_BUCKET_NAME', 'template-a-pafw-sftp-myproject-qhp0zw1')
#         else:
#             self.s3_bucket = s3_bucket

#         self.base_s3_key = os.environ.get('TTS_BASE_S3_KEY', 's3-bucket-test-isiea/dataset')
        
#         print(f"Using S3 bucket: {self.s3_bucket} with base key: {self.base_s3_key}")

#         self.voice_dict = {
#             1: f"{self.base_s3_key}/{os.environ['TTS_MALE_VOICE_1']}",
#             2: f"{self.base_s3_key}/{os.environ['TTS_MALE_VOICE_2']}",
#             # 3: f"{self.base_s3_key}/{os.environ['TTS_FEMALE_AMERICAN_VOICE_1']}",
#             # 4: f"{self.base_s3_key}/{os.environ['TTS_FEMALE_AMERICAN_VOICE_2']}"
#         }

#         self.sm_runtime = boto3.client(
#             "sagemaker-runtime",
#             region_name="ap-southeast-1",
#             aws_access_key_id=os.environ['AWS_SAGEMAKER_ACCESS_KEY_ID'],
#             aws_secret_access_key=os.environ['AWS_SAGEMAKER_SECRET_ACCESS_KEY']
#         )

#         self.s3= boto3.client(
#             "s3", 
#             region_name="ap-southeast-1",
#             config=Config(signature_version='s3v4'),
#             aws_access_key_id=os.environ['AWS_SAGEMAKER_ACCESS_KEY_ID'],
#             aws_secret_access_key=os.environ['AWS_SAGEMAKER_SECRET_ACCESS_KEY']
#         )

#     def invoke(self, text: str, voice_key: None):
#         # Build payload with proper structure
#         if voice_key is None:
#             s3_audio_prompt_path = self.generate_presigned_url(self.s3_bucket, f"{self.base_s3_key}/sample-audio-male-2.wav")
#         else:
#             s3_voice_key = self.voice_dict.get(voice_key, self.voice_dict[1])
#             s3_audio_prompt_path = self.generate_presigned_url(self.s3_bucket, s3_voice_key)

#         payload = {
#                 "text": text,
#                 "parameters": {
#                     "audio_prompt_url": s3_audio_prompt_path
#                 }
#             }
#         try:
#             # Invoke endpoint with Accept header for WAV
#             response = self.sm_runtime.invoke_endpoint(
#                 EndpointName=self.endpoint_name,
#                 ContentType='application/json',
#                 Accept='audio/wav',
#                 Body=json.dumps(payload).encode('utf-8')
#             )
            
#             # Handle binary audio response
#             audio_data = response['Body'].read()
#             return audio_data
                
#         except Exception as e:
#             print(f"✗ Error: {e}")
#             import traceback
#             traceback.print_exc()
#             return None

#     def upload_audio_prompt(self, local_audio_path: str, s3_key: str = None) -> str:
#         """Upload audio prompt to S3 and return S3 URI"""
#         if s3_key is None:
#             s3_key = f"{self.base_s3_key}/{os.path.basename(local_audio_path)}"
#         else:
#             s3_key = f"{self.base_s3_key}/{s3_key}"
            
#         try:
#             self.s3.upload_file(local_audio_path, self.s3_bucket, s3_key)
#             s3_uri = f"s3://{self.s3_bucket}/{s3_key}"
#             logging.info(f"Uploaded audio prompt to {s3_uri}")
#             return s3_uri
#         except (ClientError, FileNotFoundError) as e:
#             logging.error(f"Audio upload error: {e}")
#             raise
    
#     def generate_presigned_url(self, bucket_name, object_key, expiration=3600):
#             try:
#                 url = self.s3.generate_presigned_url(
#                     'get_object',
#                     Params={'Bucket': bucket_name, 'Key': object_key},
#                     ExpiresIn=expiration
#                 )
#                 return url
#             except Exception as e:
#                 print(f"Error generating presigned URL: {e}")
#                 return None



class SageMakerTTSClient:
    """Base class for SageMaker TTS clients with shared AWS configuration"""
    
    def __init__(self, endpoint_name: str = None, s3_bucket: str = None):
        # AWS Region
        self.region_name = "ap-southeast-1"
        
        # Set S3 bucket for inputs/outputs
        if s3_bucket is None:
            self.s3_bucket = os.environ.get('TTS_BUCKET_NAME', 'template-a-pafw-sftp-myproject-qhp0zw1')
        else:
            self.s3_bucket = s3_bucket

        self.base_s3_key = os.environ.get('TTS_BASE_S3_KEY', 's3-bucket-test-isiea/dataset')
        
        print(f"Using S3 bucket: {self.s3_bucket} with base key: {self.base_s3_key}")

        # Initialize SageMaker runtime client
        self.sm_runtime = boto3.client(
            "sagemaker-runtime",
            region_name=self.region_name,
            aws_access_key_id=os.environ['AWS_SAGEMAKER_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['AWS_SAGEMAKER_SECRET_ACCESS_KEY']
        )

        # Initialize S3 client
        self.s3 = boto3.client(
            "s3", 
            region_name=self.region_name,
            config=Config(signature_version='s3v4'),
            aws_access_key_id=os.environ['AWS_SAGEMAKER_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['AWS_SAGEMAKER_SECRET_ACCESS_KEY']
        )

        # Store endpoint name (to be set by subclasses)
        self.endpoint_name = endpoint_name

    def generate_presigned_url(self, bucket_name, object_key, expiration=3600):
        """Generate presigned URL for S3 objects"""
        try:
            url = self.s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket_name, 'Key': object_key},
                ExpiresIn=expiration
            )
            return url
        except Exception as e:
            print(f"Error generating presigned URL: {e}")
            return None
        
    def invoke(self, text: str, **kwargs):
        """Abstract method to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement the invoke method")


class ChatterboxTTSClient(SageMakerTTSClient):
    """Chatterbox TTS client using regular SageMaker endpoint invocation"""
    
    def __init__(self, endpoint_name: str = None, s3_bucket: str = None):
        super().__init__(endpoint_name, s3_bucket)
        
        # Set default endpoint if not provided
        if self.endpoint_name is None:
            self.endpoint_name = os.environ['TTS_ENDPOINT']
        
        # Voice dictionary for Chatterbox
        self.voice_dict = {
            1: f"{self.base_s3_key}/{os.environ['TTS_MALE_VOICE_1']}",
            2: f"{self.base_s3_key}/{os.environ['TTS_MALE_VOICE_2']}",
            # 3: f"{self.base_s3_key}/{os.environ['TTS_FEMALE_AMERICAN_VOICE_1']}",
            # 4: f"{self.base_s3_key}/{os.environ['TTS_FEMALE_AMERICAN_VOICE_2']}"
        }

    def invoke(self, text: str, voice_key: int = None):
        """
        Invoke Chatterbox TTS endpoint
        
        Args:
            text (str): Text to convert to speech
            voice_key (int): Voice key (1 or 2), defaults to voice 2 if None
            
        Returns:
            bytes: WAV audio data or None if failed
        """
        # Build payload with proper structure
        if voice_key is None:
            s3_audio_prompt_path = self.generate_presigned_url(
                self.s3_bucket, 
                f"{self.base_s3_key}/sample-audio-male-2.wav"
            )
        else:
            s3_voice_key = self.voice_dict.get(voice_key, self.voice_dict[1])
            s3_audio_prompt_path = self.generate_presigned_url(self.s3_bucket, s3_voice_key)

        payload = {
            "text": text,
            "parameters": {
                "audio_prompt_url": s3_audio_prompt_path
            }
        }
        
        try:
            print(f"[DEBUG TTS] Using voice prompt URL: {s3_audio_prompt_path}")
            
            # Invoke endpoint with Accept header for WAV
            response = self.sm_runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='application/json',
                Accept='audio/wav',
                Body=json.dumps(payload).encode('utf-8')
            )
            
            # Handle binary audio response
            audio_data = response['Body'].read()
            print(f"[DEBUG TTS] Received {len(audio_data)} bytes")
            
            return audio_data
                
        except Exception as e:
            print(f"✗ Chatterbox TTS Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def upload_audio_prompt(self, local_audio_path: str, s3_key: str = None) -> str:
        """Upload audio prompt to S3 and return S3 URI"""
        if s3_key is None:
            s3_key = f"{self.base_s3_key}/{os.path.basename(local_audio_path)}"
        else:
            s3_key = f"{self.base_s3_key}/{s3_key}"
            
        try:
            self.s3.upload_file(local_audio_path, self.s3_bucket, s3_key)
            s3_uri = f"s3://{self.s3_bucket}/{s3_key}"
            logging.info(f"Uploaded audio prompt to {s3_uri}")
            return s3_uri
        except (ClientError, FileNotFoundError) as e:
            logging.error(f"Audio upload error: {e}")
            raise


class KyutaiTTSClient(SageMakerTTSClient):
    """Kyutai TTS client using streaming SageMaker endpoint"""
    
    def __init__(self, endpoint_name: str = None, s3_bucket: str = None):
        super().__init__(endpoint_name, s3_bucket)
        
        # Set default endpoint if not provided
        if self.endpoint_name is None:
            self.endpoint_name = os.environ.get('KYUTAI_TTS_ENDPOINT', 'kyutai-tts-2025-07-24-08-06-16')

        # Voice dictionary for Kyutai TTS
        self.voice_dict = {
            1: "vctk/p374_023.wav",
            2: "vctk/p280_023.wav", 
            # Add more voice IDs as needed but currently only 2 are available as script is generated with speaker ID 1 and 2 only
            # This means that speaker ID 1 uses voice ID 1 and speaker ID 2 uses voice ID 2
            # If you want to add more voices, you can add them here but you will need to update the script generation logic accordingly
        }

    def invoke(self, text: str, voice_key: int = None):
        """
        Invoke Kyutai TTS streaming endpoint
        
        Args:
            text (str): Text to convert to speech
            voice_url (str): Optional voice cloning URL
            return_bytes (bool): If True, returns bytes; if False, saves to file and returns filepath
            
        Returns:
            bytes or str: WAV audio data (if return_bytes=True) or filepath (if return_bytes=False)
        """
        # Prepare payload
        payload = {"text": text}
        
        if voice_key is not None:
            voice_id = self.voice_dict.get(voice_key)
            if voice_id:
                payload["voice_id"] = voice_id
                print(f"Using voice ID {voice_key}: {voice_id}")
            else:
                print(f"Warning: Voice key {voice_key} not found, using default voice")
                print(f"Available voice keys: {list(self.voice_dict.keys())}")
        else:
            print("Using default Kyutai voice")

        try:
            print(f"Calling Kyutai streaming endpoint: {self.endpoint_name}")
            
            # Invoke streaming endpoint
            response = self.sm_runtime.invoke_endpoint_with_response_stream(
                EndpointName=self.endpoint_name,
                Body=json.dumps(payload),
                ContentType="application/json"
            )
            
            print("Connection successful. Receiving audio stream...")
            event_stream = response['Body']
            
            # Collect all audio chunks
            audio_chunks = []
            chunk_count = 0
            
            for event in event_stream:
                if 'PayloadPart' in event:
                    chunk = event['PayloadPart']['Bytes']
                    audio_chunks.append(chunk)
                    chunk_count += 1
                    
                    # Progress indicator for long streams
                    if chunk_count % 10 == 0:
                        print(f"Received {chunk_count} chunks...")
            
            # Process audio
            if audio_chunks:
                print(f"Stream finished. Received {len(audio_chunks)} chunks total.")
                
                # Concatenate all chunks
                full_audio_data = b''.join(audio_chunks)
                print(f"Total audio data size: {len(full_audio_data)} bytes")
                
                # Convert raw audio to proper WAV format with headers
                return self._add_wav_headers(full_audio_data)

            else:
                print("⚠️ No audio data received from the stream.")
                return None
                
        except Exception as e:
            print(f"✗ Kyutai TTS Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _add_wav_headers(self, raw_audio_data: bytes) -> bytes:
        """Add WAV headers to raw audio data"""
        import io
        
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(1)      # Mono
            wf.setsampwidth(2)      # 16-bit
            wf.setframerate(24000)  # 24kHz sample rate
            wf.writeframes(raw_audio_data)
        
        return wav_buffer.getvalue()

