"""
Test version of codeAssigner with batch concurrency limiting.
This adds a semaphore to limit how many batches run concurrently.
"""

# Copy all imports and most code from codeAssigner.py, just modify _process_all_batches
import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

import asyncio
import nest_asyncio
import time
from typing import Dict, List, Optional
import instructor
from openai import AsyncOpenAI
import tiktoken
from pydantic import BaseModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import OPENAI_API_KEY, DEFAULT_LANGUAGE, ModelConfig, CodeAssignmentConfig, DEFAULT_CODE_ASSIGNMENT_CONFIG, EmbeddingConfig
from prompts import CODE_ASSIGNMENT_PROMPT
import models
from .verboseReporter import VerboseReporter
from utils.embedder import Embedder

# Import the complete CodeAssigner class and modify only the problematic method
from .codeAssigner import CodeAssigner, CodeAssignmentResponse

class CodeAssignerLimited(CodeAssigner):
    """
    Test version of CodeAssigner with limited batch concurrency.
    Only the _process_all_batches method is modified to add concurrency limiting.
    """
    
    def __init__(self, *args, max_concurrent_batches: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_concurrent_batches = max_concurrent_batches
        print(f"[LIMITED] Max concurrent batches set to: {max_concurrent_batches}")
    
    async def _process_all_batches(self, batches: List[List[tuple]]) -> List[CodeAssignmentResponse]:
        """Process all batches with LIMITED concurrency to prevent bottlenecks"""
        total_ideas = sum(len(batch) for batch in batches)
        
        # Calculate total sub-batches for reporting
        total_sub_batches = sum(len(self._create_sub_batches(batch, sub_batch_size=5)) for batch in batches)
        
        self.verbose_reporter.stat_line(
            f"Processing {total_ideas} ideas in {len(batches)} batches "
            f"({total_sub_batches} concurrent sub-batches)..."
        )
        
        print(f"\n[LIMITED] Processing {len(batches)} batches with MAX {self.max_concurrent_batches} concurrent")
        print(f"[LIMITED] This limits maximum concurrent operations to prevent system overload")
        
        # SEMAPHORE TO LIMIT BATCH CONCURRENCY
        batch_semaphore = asyncio.Semaphore(self.max_concurrent_batches)
        
        async def process_batch_with_limit(batch: List[tuple], batch_index: int) -> List[CodeAssignmentResponse]:
            async with batch_semaphore:
                return await self._process_batch(batch, batch_index)
        
        # Create batch tasks with concurrency limiting
        batch_tasks = [process_batch_with_limit(batch, i) for i, batch in enumerate(batches)]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Collect results from all batches
        all_results = []
        total_failures = 0
        
        for i, batch_result in enumerate(batch_results):
            if isinstance(batch_result, Exception):
                print(f"Batch {i+1} processing failed completely: {str(batch_result)}")
                total_failures += 1
                continue
            
            # Add all results from this batch
            all_results.extend(batch_result)
        
        if total_failures > 0:
            print(f"{total_failures} out of {len(batches)} batches failed completely")
        
        print(f"[LIMITED] Completed processing with {self.max_concurrent_batches} batch limit")
        
        return all_results