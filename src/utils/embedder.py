import os
import numpy as np
import asyncio
from typing import List, Dict
from openai import AsyncOpenAI
import models
from config import OPENAI_API_KEY, DEFAULT_EMBEDDING_MODEL

class Embedder:
    def __init__(
        self, 
        provider: str = "openai",
        client: any = None, 
        embedding_model: str = None, 
        var_lab: str = None):
        
        self.client = client or AsyncOpenAI(api_key=os.getenv(OPENAI_API_KEY))
        self.embedding_model = embedding_model or DEFAULT_EMBEDDING_MODEL
        self.var_lab = var_lab
        
        print(f"Initialized Embedder with {provider} provider and {self.embedding_model} model")
    
    async def generate_description_batches(self, data: List[models.EmbeddingsModel], batch_size: int = 100):
        all_segments = []
        segment_indices = []
        
        if self.var_lab:
            for resp_idx, resp_item in enumerate(data):
                for seg_idx, segment in enumerate(resp_item.response_segment):
                    text_to_embed = self.var_lab + segment.code_description
                    all_segments.append(text_to_embed)
                    segment_indices.append((resp_idx, seg_idx))
        else:
            for resp_idx, resp_item in enumerate(data):
                for seg_idx, segment in enumerate(resp_item.response_segment):
                    text_to_embed = segment.code_description
                    all_segments.append(text_to_embed)
                    segment_indices.append((resp_idx, seg_idx))
        
        
        batches_responses = [all_segments[i:i+batch_size] for i in range(0, len(all_segments), batch_size)]
        batches_indices = [segment_indices[i:i+batch_size] for i in range(0, len(segment_indices), batch_size)] 
        
        return batches_responses, batches_indices
        
    async def generate_code_batches(self, data: List[models.EmbeddingsModel], batch_size: int = 100):
        all_segments = []
        segment_indices = []
        
        if self.var_lab:
            for resp_idx, resp_item in enumerate(data):
                for seg_idx, segment in enumerate(resp_item.response_segment):
                    text_to_embed = self.var_lab + segment.descriptive_code.replace("_", " ").title() # Code syntax (capitals + underscore) needs to be converted
                    all_segments.append(text_to_embed)
                    segment_indices.append((resp_idx, seg_idx))
        else:
            for resp_idx, resp_item in enumerate(data):
                for seg_idx, segment in enumerate(resp_item.response_segment):
                    text_to_embed = segment.descriptive_code.replace("_", " ").title() # Code syntax (capitals + underscore) needs to be converted
                    all_segments.append(text_to_embed)
                    segment_indices.append((resp_idx, seg_idx))
    
        batches_responses = [all_segments[i:i+batch_size] for i in range(0, len(all_segments), batch_size)]
        batches_indices = [segment_indices[i:i+batch_size] for i in range(0, len(segment_indices), batch_size)]
        
        return batches_responses, batches_indices

    async def process_batches(self, responses: List[str]):
        response = await self.client.embeddings.create(input=responses, model=self.embedding_model)
        return [item.embedding for item in response.data]
        
    def get_code_embeddings(self, data: List[models.EmbeddingsModel], var_lab: str = None, max_concurrent: int = 5):
        if var_lab is not None:
            self.var_lab = var_lab
            
        return asyncio.run(self._get_code_embeddings_async(data, max_concurrent))
      
    def get_description_embeddings(self, data: List[models.EmbeddingsModel], var_lab: str = None, max_concurrent: int = 5):
        if var_lab is not None:
            self.var_lab = var_lab
            
        return asyncio.run(self._get_description_embeddings_async(data, max_concurrent))
        
    async def _get_description_embeddings_async(self, data: List[models.EmbeddingsModel], max_concurrent: int = 5):
        batches_responses, batches_indices = await self.generate_description_batches(data)
        return await self._process_embeddings(data, batches_responses, batches_indices, max_concurrent, is_description=True)
    
    async def _get_code_embeddings_async(self, data: List[models.EmbeddingsModel], max_concurrent: int = 5):
        batches_responses, batches_indices = await self.generate_code_batches(data)
        return await self._process_embeddings(data, batches_responses, batches_indices, max_concurrent, is_description=False)
    
    async def _process_embeddings(self, data: List[models.EmbeddingsModel], batches_responses, batches_indices, max_concurrent: int = 5, is_description: bool = False):
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_batch(batch_responses, batch_indices):
            async with semaphore:
                #print(f"Processing batch with {len(batch_responses)} items")
                batch_embeddings = await self.process_batches(batch_responses)
                
                for (resp_idx, seg_idx), embedding in zip(batch_indices, batch_embeddings):
                    # save as array
                    embedding_array = np.array(embedding, dtype=np.float32)
                    
                    if is_description:
                        data[resp_idx].response_segment[seg_idx].description_embedding = embedding_array
                    else:
                        data[resp_idx].response_segment[seg_idx].code_embedding = embedding_array
                    
                #print(f"Completed batch with {len(batch_responses)} items")

        # Create and run tasks
        tasks = [
            process_batch(batch_responses, batch_indices) 
            for batch_responses, batch_indices in zip(batches_responses, batches_indices)]
        
        print(f"Starting to process {len(tasks)} batches with max {max_concurrent} concurrent requests")
        await asyncio.gather(*tasks)
        print(f"Completed all {len(tasks)} batches")
        
        return data
    
    def combine_embeddings(self, code_embeddings: list, description_embeddings: list) -> list:
       
        # Create a dictionary to easily look up description embeddings by respondent_id and segment_id
        description_map = {}
        for resp in description_embeddings:
            resp_id = resp.respondent_id
            description_map[resp_id] = {}
            for segment in resp.response_segment:
                seg_id = segment.segment_id
                description_map[resp_id][seg_id] = segment.description_embedding
        
        # Start with the code embeddings as the base
        combined_embeddings = code_embeddings.copy()  # Make a shallow copy to avoid modifying original
        
        # Add description embeddings where they exist
        for resp in combined_embeddings:
            resp_id = resp.respondent_id
            if resp_id in description_map:
                for segment in resp.response_segment:
                    seg_id = segment.segment_id
                    if seg_id in description_map[resp_id]:
                        segment.description_embedding = description_map[resp_id][seg_id]
        
        return combined_embeddings

    async def _async_embed_word_clusters(self, clusters: Dict[int, List[str]]) -> np.ndarray:
   
        client = self.client 
        
        cluster_texts = []
        for cluster_id, words in clusters.items():
            full_cluster_text = " ".join(words)
            cluster_texts.append(full_cluster_text)
        
        tasks = []
        for cluster_text in cluster_texts:
            task = client.embeddings.create(
                input=cluster_text,
                model="text-embedding-3-large"
            )
            tasks.append(task)
    
        responses = await asyncio.gather(*tasks)
    
        embeddings = [response.data[0].embedding for response in responses]
        return np.array(embeddings)
    
    def embed_word_clusters(self, clusters: Dict[int, List[str]]) -> np.ndarray: 
        return asyncio.run(self._async_embed_word_clusters(clusters))
    
    async def _async_embed_words(self, words: List[str], batch_size: int = 100, max_concurrent: int = 5) -> Dict[str, np.ndarray]:
        semaphore = asyncio.Semaphore(max_concurrent)
        result_dict = {}
        
        word_batches = [words[i:i+batch_size] for i in range(0, len(words), batch_size)]
        
        async def process_batch(batch_words):
            async with semaphore:
                print(f"Processing batch of {len(batch_words)} words")
                response = await self.client.embeddings.create(
                    input=batch_words,
                    model=self.embedding_model
                )
                
                # Map each word to its embedding
                for i, item in enumerate(response.data):
                    word = batch_words[i]
                    embedding = np.array(item.embedding, dtype=np.float32)
                    result_dict[word] = embedding
                
                print(f"Completed batch of {len(batch_words)} words")
        
        # Create and run tasks
        tasks = [process_batch(batch) for batch in word_batches]
        
        print(f"Starting to process {len(tasks)} batches with max {max_concurrent} concurrent requests")
        await asyncio.gather(*tasks)
        print(f"Completed embeddings for {len(words)} words")
        
        ordered_embeddings = [result_dict[word] for word in words]
        
        return ordered_embeddings
    
    def embed_words(self, words: List[str], batch_size: int = 100, max_concurrent: int = 5) -> Dict[str, np.ndarray]:
       
        return asyncio.run(self._async_embed_words(words, batch_size, max_concurrent))



# Example usage
if __name__ == "__main__":
    
    from utils import data_io, csvHandler
    
    filename                = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
    id_column               = "DLNMID"
    var_name                = "Q20"

    csv_handler             = csvHandler.CsvHandler()
    filepath                = csv_handler.get_filepath(filename, 'segmented_descriptions')
    data_loader             = data_io.DataLoader()
    var_lab                 = data_loader.get_varlab(filename=filename, var_name=var_name)

    segmented_text          = csv_handler.load_from_csv(filename, 'segmented_descriptions', models.DescriptiveModel)
    input_data              = [item.to_model(models.EmbeddingsModel) for item in segmented_text]
    
    for response in input_data:
        print(f"\nRespondent ID: {response.respondent_id}")
        print(f"Response: {response.response}")
        print("Descriptive Codes:")
        response_segments = response.response_segment or []
        for segment in response_segments:
            print(f"  - Segment: {segment.segment_response}")
            print(f"    Code: {segment.descriptive_code}")
            
            print(segment.descriptive_code.replace("_", " ").title())
            
            print(f"    Description: {segment.code_description}")
    
    embedder                = Embedder()
    
    code_embeddings        = embedder.get_code_embeddings(input_data)
    description_embeddings = embedder.get_description_embeddings(input_data, var_lab)
    combined_embeddings    = embedder.combine_embeddings(code_embeddings, description_embeddings)

    for response in combined_embeddings[:1]:
        print(f"\nRespondent ID: {response.respondent_id}")
        print(f"Response: {response.response}")
        print("Descriptive Codes:")
        response_segments = response.response_segment or []
        for segment in response_segments:
            print(f"  - Segment: {segment.segment_response}")
            print(f"    Code: {segment.descriptive_code}")
            print(f"    Description: {segment.code_description}")
            print(f"    Description: {segment.code_embedding}")
            print(f"    Description: {segment.description_embedding}")
        print("\n")
              
    
        
    
    
    
