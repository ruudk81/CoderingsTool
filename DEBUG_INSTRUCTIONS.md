# 🔍 codeAssigner Bottleneck Debugging Instructions

## What's Changed

The main `codeAssigner.py` file now includes comprehensive debugging tools that will **automatically** activate when you run your pipeline. No code changes needed!

## What Will Happen When You Run the Pipeline

1. **Debugging Activation**: When codeAssigner initializes, you'll see:
   ```
   🔍 DEBUGGING MODE: Concurrency tracking enabled
   ```

2. **Real-time Monitoring**: During processing, you'll see:
   ```
   🔍 DEBUG: About to process ALL X batches concurrently
   🔍 DEBUG: This means up to Y sub-batches could run simultaneously  
   🔍 DEBUG: With 5 API calls per sub-batch, that's up to Z concurrent API calls!
   ```

3. **Comprehensive Report**: After processing completes, you'll get a detailed report showing:
   - **Exact concurrency numbers** (how many batches/API calls ran simultaneously)
   - **Resource usage** (memory, CPU)
   - **Performance metrics** (timing breakdown)
   - **Potential bottlenecks** (automatically detected)
   - **Timeline data** (saved to JSON file for analysis)

## Testing the "Too Many Batches" Hypothesis

If you want to test limiting batch concurrency:

1. **Edit lines 506-511** in `codeAssigner.py`
2. **Uncomment those lines** to enable batch limiting:
   ```python
   max_concurrent_batches = 10  # Test with limited concurrency
   batch_semaphore = asyncio.Semaphore(max_concurrent_batches)
   async def process_batch_limited(batch, i):
       async with batch_semaphore:
           return await self._process_batch(batch, i)
   batch_tasks = [process_batch_limited(batch, i) for i, batch in enumerate(batches)]
   ```

3. **Run your pipeline** - if it's much faster, we've confirmed the hypothesis!

## What to Look For

The debugging will reveal:

- **High concurrent API calls** (>100): Likely hitting rate limits
- **High memory usage** (>1GB): Resource exhaustion
- **Many concurrent batches** (>50): System overload
- **Long API call durations**: Network/API issues

## Files Created

- `codeAssigner_original.py` - Your original working version (backup)
- `codeAssigner_debug_timeline_XXXXX.json` - Detailed timeline data
- Debug output will show in console during pipeline run

## Next Steps

1. **Run your pipeline** with 1000+ ideas
2. **Check the debug report** at the end
3. **If bottleneck confirmed**, we'll implement the fix
4. **If not**, we'll investigate other possibilities

The debugging will give us **complete certainty** about what's causing the bottleneck!