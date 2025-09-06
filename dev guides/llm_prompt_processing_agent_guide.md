# LLM Prompt Processing Agent Guide

## Overview

This guide provides instructions for implementing LLM prompt processing logic that aligns with CoderingsTool's proven design patterns. The approach is based on the production-tested spellChecker.py implementation and follows the architectural principles established in the project.

## Core Design Philosophy

### 1. Prompts Lead the Design

**Principle**: Pydantic models and processing logic should be designed to facilitate prompted outputs from LLMs.

```python
# Design flow: Prompt → Pydantic Model → Processing Logic
# NOT: Processing Logic → Prompt → Model (retrofitted)
```

**Implementation Pattern**:
1. **Start with prompt design** - what structured output do you need?
2. **Create Pydantic models** that match the prompt's expected JSON structure exactly
3. **Build processing logic** that handles the validated model outputs

### 2. Structured Data First

All LLM interactions must use structured outputs with Pydantic validation:

```python
from pydantic import BaseModel
from typing import List, Optional

class YourTaskResponse(BaseModel):
    """Model that exactly matches your prompt's expected JSON structure"""
    task_id: str
    result: str
    confidence: Optional[float] = None
    rationale: Optional[str] = None

class YourBatchResponse(BaseModel):
    """Wrapper for multiple task responses"""
    responses: List[YourTaskResponse]
    processing_metadata: Optional[Dict[str, Any]] = None
```

### 3. Individual Task Processing with Rate Limiting

Follow the proven three-layer approach from spellChecker.py:

```python
async def process_individual_task(self, task):
    tokens_needed = self.count_task_tokens(task)
    
    # Three-layer rate limiting (EXACT ORDER)
    async with self.rpm_limiter:                    # 1. RPM first
        await self.token_bucket.acquire(tokens_needed)  # 2. TPM second  
        async with self.semaphore:                     # 3. Transport last
            return await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    response_model=YourTaskResponse,  # Pydantic validation
                    messages=self.create_messages(task),
                    temperature=self.config.temperature,
                    max_retries=0
                ),
                timeout=15
            )
```

## Implementation Template

### Step 1: Define Your Prompt Structure

```python
# prompts.py
YOUR_TASK_INSTRUCTIONS = """
You are a {language} expert performing {task_type} on survey responses.

Task Description:
{task_description}

Survey Question:
<survey_question>
{var_lab}
</survey_question>

Process these tasks:
<tasks>
{tasks}
</tasks>

Output Format:
Provide your results as valid JSON matching this exact structure:
{{
  "responses": [
    {{
      "task_id": "ID_FROM_TASK",
      "result": "Your result here",
      "confidence": 0.95,
      "rationale": "Brief explanation of your reasoning"
    }},
    ...
  ]
}}

Guidelines:
- Follow the survey context when making decisions
- Provide confidence scores between 0.0 and 1.0
- Include brief rationale for complex cases
- Ensure JSON is properly formatted
"""
```

### Step 2: Create Aligned Pydantic Models

```python
# models.py - Add to your existing models
class YourTaskItem(BaseModel):
    task_id: Any  # Match CoderingsTool pattern (Any for flexibility)
    original_input: str
    processed_output: Optional[str] = None
    confidence: Optional[float] = None
    rationale: Optional[str] = None
    
class YourTaskResponse(BaseModel):
    """Single task response - matches prompt JSON structure exactly"""
    task_id: Any
    result: str
    confidence: float
    rationale: Optional[str] = None

class YourBatchResponse(BaseModel):
    """Batch response wrapper - matches prompt JSON structure exactly"""
    responses: List[YourTaskResponse]
```

### Step 3: Implement Processing Class

```python
import asyncio
import time
import logging
from typing import List, Dict, Any, Optional
from aiolimiter import AsyncLimiter
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter
from openai import RateLimitError

from .cached_resources import get_openai_client, get_tiktoken_encoding
from .verboseReporter import VerboseReporter
from config import get_openai_rate_limits, ModelConfig
from prompts import YOUR_TASK_INSTRUCTIONS

class YourProcessor:
    def __init__(self, config=None, model_config: ModelConfig = None, 
                 openai_api_key: Optional[str] = None, verbose: bool = False, 
                 verbose_reporter: Optional[VerboseReporter] = None):
        
        self.config = config or YourProcessorConfig()  # Define your config class
        self.model_config = model_config or ModelConfig()
        self.model = self.model_config.get_model_for_stage('your_stage_name')
        
        # Instructor-patched client for structured output
        self.client = get_openai_client(openai_api_key)
        self.verbose_reporter = verbose_reporter or VerboseReporter(verbose)
        
        # Rate limiting setup (follows spellChecker.py pattern exactly)
        limits = get_openai_rate_limits(self.model)
        HEADROOM = 0.8
        
        self.rpm_limiter = AsyncLimiter(limits.requests_per_minute * HEADROOM / 60, 1)
        self.token_bucket = TokenBucket(limits.tokens_per_minute * HEADROOM)
        self.semaphore = asyncio.Semaphore(100)
        
        # Stats tracking (CoderingsTool pattern)
        self.stats = {
            'tasks_processed': 0,
            'tasks_successful': 0,
            'tasks_failed': 0,
            'llm_calls_made': 0,
            'llm_calls_successful': 0,
            'llm_calls_failed': 0,
            'processing_time': 0.0
        }
    
    def count_task_tokens(self, task: Dict[str, Any]) -> int:
        """Count tokens with output estimates (critical for TPM limiting)"""
        # Create the actual prompt you'll send
        task_text = f"""Task ID: {task['task_id']}
Input: {task['input_text']}
Additional context: {task.get('context', '')}"""
        
        full_prompt = YOUR_TASK_INSTRUCTIONS.format(
            language=self.config.language,
            task_type=self.config.task_type,
            task_description=self.config.task_description,
            var_lab=self.config.var_lab,
            tasks=task_text
        )
        
        encoding = get_tiktoken_encoding(self.model)
        input_tokens = len(encoding.encode(full_prompt))
        
        # Estimate output tokens based on your task type
        # Adjust this ratio based on your specific use case:
        # - Classification: 10-20% of input
        # - Correction: 30-50% of input  
        # - Generation: 100-200% of input
        estimated_output_tokens = max(50, int(input_tokens * 0.3))
        
        return input_tokens + estimated_output_tokens
    
    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential_jitter(initial=1, max=30),
        stop=stop_after_attempt(3),
        reraise=True
    )
    async def process_individual_task(self, task: Dict[str, Any], var_lab: str) -> Dict[str, Any]:
        """Process single task following CoderingsTool patterns"""
        
        tokens_needed = self.count_task_tokens(task)
        
        # Three-layer rate limiting (EXACT ORDER from spellChecker.py)
        async with self.rpm_limiter:                    # 1. RPM check first
            await self.token_bucket.acquire(tokens_needed)  # 2. TPM check second
            async with self.semaphore:                     # 3. Transport limit last
                
                try:
                    self.stats['llm_calls_made'] += 1
                    
                    # Create prompt from task
                    task_text = f"""Task ID: {task['task_id']}
Input: {task['input_text']}
Context: {task.get('context', '')}"""
                    
                    full_prompt = YOUR_TASK_INSTRUCTIONS.format(
                        language=self.config.language,
                        task_type=self.config.task_type,
                        task_description=self.config.task_description,
                        var_lab=var_lab,
                        tasks=task_text
                    )
                    
                    # Use instructor client for structured output (CRITICAL)
                    response = await asyncio.wait_for(
                        self.client.chat.completions.create(
                            model=self.model,
                            response_model=YourBatchResponse,  # Pydantic validation
                            messages=[{"role": "user", "content": full_prompt}],
                            temperature=self.config.temperature,
                            seed=self.config.seed,
                            max_retries=0  # Let tenacity handle retries
                        ),
                        timeout=15  # Prevent stragglers
                    )
                    
                    self.stats['llm_calls_successful'] += 1
                    
                    # Extract result from structured response
                    if response.responses and len(response.responses) > 0:
                        result = response.responses[0]
                        return {
                            'task_id': task['task_id'],
                            'original_input': task['input_text'],
                            'processed_output': result.result,
                            'confidence': result.confidence,
                            'rationale': result.rationale
                        }
                    else:
                        # Fallback - no valid response
                        return self.create_fallback_response(task)
                        
                except asyncio.TimeoutError:
                    logging.warning(f"Task {task['task_id']} timed out after 15s")
                    self.stats['llm_calls_failed'] += 1
                    return self.create_fallback_response(task)
                    
                except Exception as e:
                    logging.error(f"Task {task['task_id']} failed: {e}")
                    self.stats['llm_calls_failed'] += 1
                    return self.create_fallback_response(task)
    
    def create_fallback_response(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback response for failed tasks"""
        return {
            'task_id': task['task_id'],
            'original_input': task['input_text'],
            'processed_output': task['input_text'],  # Return original
            'confidence': 0.0,
            'rationale': "Processing failed - returned original input"
        }
    
    async def process_all_tasks_async(self, tasks: List[Dict[str, Any]], var_lab: str) -> List[YourTaskItem]:
        """Main processing method following CoderingsTool patterns"""
        
        if not tasks:
            return []
            
        # Setup and reporting (follows spellChecker.py style)
        print(f"Processing {len(tasks)} tasks...")
        
        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line(f"Model: {self.model}")
            self.verbose_reporter.stat_line(f"Tasks to process: {len(tasks)}")
        
        # Create individual task coroutines (NOT batches)
        task_coroutines = [
            self.process_individual_task(task, var_lab) 
            for task in tasks
        ]
        
        # Process with protected gathering (CRITICAL)
        print(f"Processing tasks... 0/{len(tasks)} (0.0%)")
        start_time = time.time()
        
        results = await asyncio.gather(*task_coroutines, return_exceptions=True)
        
        processing_time = time.time() - start_time
        
        # Handle results safely (follows spellChecker.py pattern)
        processed_results = []
        successful_tasks = 0
        failed_tasks = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logging.error(f"Task {i} failed with exception: {result}")
                fallback = self.create_fallback_response(tasks[i])
                processed_results.append(YourTaskItem(**fallback))
                failed_tasks += 1
            else:
                processed_results.append(YourTaskItem(**result))
                successful_tasks += 1
        
        # Statistics and reporting (CoderingsTool style)
        success_rate = (successful_tasks / len(tasks)) * 100
        self.stats['processing_time'] = processing_time
        
        print(f"Processing tasks... {len(tasks)}/{len(tasks)} (100.0%)")
        print(f"• Successful: {successful_tasks}")
        print(f"• Failed: {failed_tasks}")
        print(f"• Success rate: {success_rate:.1f}%")
        
        if processing_time > 1:
            rate = len(tasks) / processing_time
            print(f"• Processing rate: {rate:.1f} tasks/sec")
        
        return processed_results
```

### Step 4: Integration with Pipeline

```python
# Add to your pipeline step
async def your_processing_step(responses, config, var_lab):
    """Pipeline step following CoderingsTool patterns"""
    
    if not responses:
        return responses
    
    processor = YourProcessor(
        config=config.your_processor_config,
        model_config=config.model_config,
        verbose_reporter=verbose_reporter
    )
    
    # Convert responses to task format
    tasks = [
        {
            'task_id': response.respondent_id,
            'input_text': response.response,
            'context': getattr(response, 'additional_context', '')
        }
        for response in responses
    ]
    
    # Process tasks
    processed_items = await processor.process_all_tasks_async(tasks, var_lab)
    
    # Convert back to pipeline format
    updated_responses = []
    for original, processed in zip(responses, processed_items):
        # Create next pipeline stage model
        updated_response = NextStageModel(
            respondent_id=original.respondent_id,
            response=original.response,
            your_new_field=processed.processed_output,
            confidence=processed.confidence,
            processing_metadata={
                'rationale': processed.rationale,
                'processing_stats': processor.stats
            }
        )
        updated_responses.append(updated_response)
    
    return updated_responses
```

## Configuration Classes

```python
# config.py - Add your processor configuration
@dataclass
class YourProcessorConfig:
    """Configuration for your processor"""
    temperature: float = 0.1
    seed: int = 42
    language: str = "Dutch"  # Or from DEFAULT_LANGUAGE
    task_type: str = "your_task_type"
    task_description: str = "Description of what your processor does"
    max_retries: int = 3
    timeout_seconds: int = 15
    
    # Task-specific settings
    confidence_threshold: float = 0.8
    enable_rationale: bool = True

# Add to ModelConfig
class ModelConfig:
    def get_model_for_stage(self, stage: str) -> str:
        stage_models = {
            'spell_check': 'gpt-4.1-mini',
            'quality_filter': 'gpt-4.1-mini', 
            'your_stage_name': 'gpt-4.1-mini',  # Add your stage
            # ... other stages
        }
        return stage_models.get(stage, DEFAULT_MODEL)
```

## Key Success Patterns

### 1. Prompt-First Design
- Design your JSON output structure first
- Create Pydantic models that match exactly
- Build processing around validated structures

### 2. Rate Limiting Compliance
- Always use the three-layer system in correct order
- Include output token estimates in counting
- Use protected gathering with exception handling

### 3. CoderingsTool Integration
- Follow existing model inheritance patterns
- Use VerboseReporter for consistent logging
- Maintain statistics tracking throughout

### 4. Error Handling
- Provide meaningful fallbacks for failed tasks
- Log failures appropriately
- Never let one failure break the entire batch

### 5. Performance Optimization
- Process individual tasks, not artificial batches
- Use asyncio.gather with return_exceptions=True
- Monitor and report processing rates

## Testing Your Implementation

```python
# test_your_processor.py
async def test_your_processor():
    """Test your processor with sample data"""
    
    processor = YourProcessor(verbose=True)
    
    test_tasks = [
        {
            'task_id': 'test_1',
            'input_text': 'Sample input for testing',
            'context': 'Test context'
        }
    ]
    
    results = await processor.process_all_tasks_async(test_tasks, "Test Variable")
    
    # Verify results
    assert len(results) == len(test_tasks)
    assert all(isinstance(r, YourTaskItem) for r in results)
    assert processor.stats['tasks_successful'] > 0
    
    print("✅ All tests passed!")

# Run test
if __name__ == "__main__":
    import asyncio
    asyncio.run(test_your_processor())
```

This guide ensures your LLM prompt processing aligns perfectly with CoderingsTool's proven architecture while maintaining the performance and reliability characteristics of the production-tested spellChecker.py implementation.