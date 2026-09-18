import torch
import time
import asyncio as aio
from pathlib import Path

from silero.model_beautified import VAD

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Init Big model
        self.conv = torch.nn.Conv1d(512, 128, kernel_size=3, padding=1)
        
    def forward(self, x):
        out = self.conv(x)
        return out

# Load or create model
loaded = torch.jit.load("assets/silero_vad.jit")
model = VAD()
model._model = loaded._model

# model = torch.jit.script(model)
# model.cpu()
model.eval()

# Warmup
print("Warming up model...")

x_warmup = torch.rand((1, 512, ))
with torch.no_grad():
    model(x_warmup, sr=16000)
print("Warmup complete")


def run_model_sync(task_id):
    """Run model synchronously in a thread"""
    start = time.time()
    with torch.no_grad():
        for i in range():
            x = torch.rand((1, 512, ))
            out = model(x, sr=16000)
    elapsed = time.time() - start
    print(f"Task {task_id}: inference took {elapsed*1000:.2f}ms, output sum: {out.sum().item():.4f}")
    return out, elapsed


async def run_async_parallel(num_tasks=10):
    """Run multiple model inferences in parallel using asyncio"""
    print(f"\n=== Running {num_tasks} parallel async tasks ===")
    tasks = []
    
    wall_start = time.time()
    
    # Create all tasks
    for i in range(num_tasks):
        task = aio.to_thread(run_model_sync, i)
        tasks.append(task)
    
    # Run all tasks in parallel
    results = await aio.gather(*tasks)
    
    wall_elapsed = time.time() - wall_start

    
    print(f"\n--- Results ---")
    print(f"Wall time: {wall_elapsed:.3f}s")

async def run_sequential(num_tasks=10):
    """Run model inferences sequentially for comparison"""
    print(f"\n=== Running {num_tasks} sequential tasks ===")
    
    wall_start = time.time()
    
    for i in range(num_tasks):
        
        await aio.to_thread(run_model_sync, i)
    
    wall_elapsed = time.time() - wall_start
    
    print(f"\n--- Results ---")
    print(f"Wall time: {wall_elapsed:.3f}s")
    

async def main():
    """Main function to test both sequential and parallel execution"""
    num_tasks = 10
    
    # Test sequential execution
    await run_sequential(num_tasks)
    
    # Test parallel execution
    await run_async_parallel(num_tasks)
    
    
if __name__ == "__main__":
    aio.run(main())
    
    