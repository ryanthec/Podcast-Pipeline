import os
import json
import re
import numpy as np
import soundfile as sf
import base64
import io
import random
from io import BytesIO
from dotenv import load_dotenv
from typing import List, Dict, Tuple
from pydub import AudioSegment
from kokoro import KPipeline
from sagemaker_client import SageMakerClient, SageMakerEmbeddingClient, SageMakerTTSClient
from rag_service import RAGService

load_dotenv()

class PodcastGenerator:
    def __init__(self, llm_endpoint: str = None):
        """Initialize the podcast generator with necessary components."""
        # Initialize SageMaker client for script generation
        self.llm_client = SageMakerClient(endpoint_name=llm_endpoint) if llm_endpoint else SageMakerClient()
        self.embedding_client = SageMakerEmbeddingClient()
        self.sagemaker_tts_client = SageMakerTTSClient()
        self.num_voices = int(os.environ['TTS_NUM_VOICES'])

        # Initialize Kokoro TTS pipeline, code a for American English
        self.tts_pipeline = KPipeline(lang_code='a')
        
        # Define speaker voices - you can customize these IDs
        self.speakers = {
            1: "af_heart",    # Female voice for Speaker 1 
            2: "am_puck"   # Male voice for Speaker 2 (good ones are puck, michael)
        }
    
    def generate_podcast_script(self, document: str) -> str:

        # Step 1: Build RAG vector store
        rag_service = RAGService(embedding_client=self.embedding_client)
        rag_service.build_vector_store(document)

        # Step 2: Retrieve relevant context (top 10 chunks)
        context = rag_service.get_context_for_llm(top_k=10)

        # Step 3: Use LLM to generate the podcast script
        system_prompt = """
            You are an expert podcast script writer. 
            Create a natural, engaging podcast script between two speakers 
            discussing the content in the document provided. The podcast should be around 5 minutes 
            when read aloud (approximately 750-900 words).

            Speaker 1 (S1) should be a host named Jordan.
            Speaker 2 (S2) should be a host named Riley.

            IMPORTANT FORMATTING RULES:
            - Use ONLY the format [S1]: for Speaker 1 and [S2]: for Speaker 2
            - Do NOT add any additional labels like "HOST 1" or "HOST 2"
            - Do NOT repeat the speaker tags
            - Start directly with [SCRIPT START] and end with [SCRIPT END]

            Example of correct formatting:

            [SCRIPT START]
            [S1]: Welcome to our podcast! I'm Jordan, and today we're exploring an amazing topic.
            [S2]: Thanks Jordan! I'm Riley, and I'm excited to dive into this with you.
            [S1]: [Speaker 1's next dialogue]
            [S2]: [Speaker 2's next dialogue]
            ...
            [SCRIPT END]

            Make the conversation sound natural, informative, and engaging. Include an introduction 
            and conclusion. The speakers should have distinct personalities - Jordan is enthusiastic 
            and asks great questions, while Riley is knowledgeable and explains concepts clearly.
            """
        
        prompt = f"Document to discuss:\n{context}\n\nCreate a podcast script."
        
        # Generate the script using SageMaker
        script = self.llm_client.prompt(prompt, system_prompt=system_prompt)
        return script
    
    def parse_script(self, script: str) -> str:
        """Parse the script to extract only the content between [SCRIPT START] and [SCRIPT END] tags."""
        
        # First, remove thinking tags and their content
        import re
        # Remove everything between <think> and </think> tags
        script_cleaned = re.sub(r'<think>.*?</think>', '', script, flags=re.DOTALL)
        
        # Extract the script between start and end tags
        start_tag = "[SCRIPT START]"
        end_tag = "[SCRIPT END]"
        
        start_index = script_cleaned.find(start_tag)
        end_index = script_cleaned.find(end_tag)
        
        if start_index != -1 and end_index != -1:
            # Extract the script between tags (excluding the tags themselves)
            extracted_script = script_cleaned[start_index + len(start_tag):end_index].strip()
            print(f"Extracted script: {extracted_script}")
            return extracted_script
        else:
            # If tags are not found, return the cleaned script
            return script_cleaned
    
    def chunk_script(self, script: str) -> List[List[Tuple[int, str]]]:
        """
        Split the script into chunks, each containing 1 utterance from each speaker.
        Returns a list of chunks, where each chunk is a list of (speaker_number, text) tuples.
        """
        # Regular expression to identify speaker tags and utterances
        pattern = r'\[S(\d+)\]\s*:\s*(.*?)(?=\[S\d+\]\s*:|$)'
        matches = re.findall(pattern, script, re.DOTALL)
        
        # Convert matches to list of (speaker_number, text) tuples
        utterances = [(int(speaker_num), text.strip()) for speaker_num, text in matches]
        
        # Group utterances into chunks (2 from speaker 1, 2 from speaker 2)
        chunks = []
        current_chunk = []
        speaker1_count = 0
        speaker2_count = 0
        
        for speaker, text in utterances:
            # Add the current utterance to the current chunk
            current_chunk.append((speaker, text))
            
            # Update the speaker count
            if speaker == 1:
                speaker1_count += 1
            elif speaker == 2:
                speaker2_count += 1
            
            # Check if we have 1 utterance from each speaker
            if speaker1_count >= 1 and speaker2_count >= 1:
                chunks.append(current_chunk)
                current_chunk = []
                speaker1_count = 0
                speaker2_count = 0
        
        # Add any remaining utterances as the last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        # Voice randomization logic
        if self.num_voices < 2:
            raise ValueError("At least 2 voices required")
        
        # Select 2 distinct random voices
        voice_keys = random.sample(range(1, self.num_voices + 1), 2)
        voice_map = {1: voice_keys[0], 2: voice_keys[1]}
        
        # Remap speaker numbers to voice keys
        remapped_chunks = []
        for chunk in chunks:
            remapped_chunk = [
                (voice_map[speaker], text) 
                for speaker, text in chunk
            ]
            remapped_chunks.append(remapped_chunk)
        
        return remapped_chunks
    
    def generate_audio_for_chunk(self, chunk: List[Tuple[int, str]]) -> np.ndarray:
        """Generate audio for a chunk of utterances and return as a single array."""
        all_audio_segments = []
        
        for speaker, text in chunk:
            # Get the appropriate voice for the speaker
            voice_id = self.speakers.get(speaker, "af_heart")  # Default to af_heart if speaker not found
            
            print(f"Generating audio for Speaker {speaker} with voice {voice_id}")
            
            # Generate speech using Kokoro
            generator = self.tts_pipeline(
                text=text,
                voice=voice_id,
                speed=1.1,
                split_pattern=r'\n+'  # Split by paragraphs for better processing
            )
            
            # Process all segments from this utterance
            utterance_segments = []
            for _, _, audio in generator:
                utterance_segments.append(audio)
            
            # Concatenate all segments for this utterance
            if utterance_segments:
                utterance_audio = np.concatenate(utterance_segments)
                all_audio_segments.append(utterance_audio)
                
                # Add a small pause after each utterance (500ms of silence at 24kHz = 12000 samples)
                pause = np.zeros(12000, dtype=np.float32)
                all_audio_segments.append(pause)
        
        # Concatenate all utterance audio segments into one chunk
        if all_audio_segments:
            return np.concatenate(all_audio_segments)
        
        return np.array([], dtype=np.float32)
    
    def generate_audio_for_chunk_sagemaker(self, chunk: list) -> np.ndarray:
        if not self.sagemaker_tts_client:
            raise ValueError("SageMaker TTS client not initialized")
            
        all_audio_segments = []
        target_sample_rate = 22050
        
        for speaker, text in chunk:
            print(f"Generating audio for Speaker {speaker} using SageMaker")
            
            try:
                # Get binary WAV data
                wav_data = self.sagemaker_tts_client.invoke(
                    text=text, 
                    voice_key=speaker
                )
                
                if wav_data:
                    # Read audio directly from bytes
                    with BytesIO(wav_data) as audio_buffer:
                        audio_array, actual_sample_rate = sf.read(
                            audio_buffer,
                            dtype='float32'
                        )
                    
                    print(f"✓ Audio received: {len(audio_array)} samples @ {actual_sample_rate}Hz")
                    
                    # Resample if necessary
                    if actual_sample_rate != target_sample_rate:
                        import librosa
                        audio_array = librosa.resample(
                            audio_array, 
                            orig_sr=actual_sample_rate, 
                            target_sr=target_sample_rate
                        )
                    
                    # Add to segments
                    all_audio_segments.append(audio_array)
                    
                    # Add 500ms pause between utterances
                    pause = np.zeros(int(0.3 * target_sample_rate), dtype=np.float32)
                    all_audio_segments.append(pause)
                else:
                    print(f"✗ Empty response for speaker {speaker}")
                    
            except Exception as e:
                print(f"Error processing speaker {speaker}: {e}")
                continue
        
        # Combine all segments
        if all_audio_segments:
            return np.concatenate(all_audio_segments)
        return np.array([], dtype=np.float32)

    def generate_podcast(self, document: str, output_path: str, model_name: str) -> str:
        """Generate a complete podcast from a document."""
        # Create outputs directory
        outputs_dir = os.path.join(os.path.dirname(output_path), "outputs")
        os.makedirs(outputs_dir, exist_ok=True)
        
        # Step 1: Generate the podcast script
        print("Generating podcast script...")
        full_script = self.generate_podcast_script(document)
        
        # Step 2: Parse the script to extract content between tags
        print("Parsing script...")
        script = self.parse_script(full_script)
        # Save the full script to a text file
        script_path = os.path.join(os.path.dirname(output_path), "script.txt")
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(full_script)
        print(f"Script saved to {script_path}")
        
        # Step 3: Chunk the script into groups of utterances
        print("Chunking script...")
        chunks = self.chunk_script(script)
        
        # Step 4: Generate audio for each chunk
        print(f"Generating audio for {len(chunks)} chunks...")
        all_chunk_files = []

        sample_rate = 24000 if model_name == "kokoro" else 22050
        
        for i, chunk in enumerate(chunks):
            print(f"\nProcessing chunk {i+1}/{len(chunks)}")
            
            if(model_name == "kokoro"):
                # Generate audio for this chunk
                chunk_audio = self.generate_audio_for_chunk(chunk)
            elif(model_name == "sagemaker"):
                # Generate audio for this chunk using SageMaker
                chunk_audio = self.generate_audio_for_chunk_sagemaker(chunk)
            else:
                raise ValueError(f"Unsupported model name: {model_name}")
            
            # Save chunk audio directly to outputs directory
            if len(chunk_audio) > 0:
                chunk_output = os.path.join(outputs_dir, f"chunk_{i+1}.wav")
                sf.write(chunk_output, chunk_audio, sample_rate)
                all_chunk_files.append(chunk_output)
                print(f"Chunk {i+1} audio saved to {chunk_output}")
        
        # Step 5: Stitch all chunks together for the final podcast
        print("Creating final podcast...")
        
        # Check if we have audio files to stitch
        if not all_chunk_files:
            print("No audio chunks were generated.")
            return None
        
        # Create a combined audio file using pydub
        combined = AudioSegment.empty()
        
        for audio_file in all_chunk_files:
            if os.path.exists(audio_file):
                segment = AudioSegment.from_wav(audio_file)
                combined += segment
        
        # Export the combined audio
        combined.export(output_path, format="wav")
        print(f"Podcast generation complete! Final output saved to {output_path}")
        
        return output_path

