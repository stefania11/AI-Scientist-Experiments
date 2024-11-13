Instructions for Running New AI-Scientist Experiments

0. Set Up Environment Variables
```bash
# Set API keys (replace with your actual keys)
export OPENAI_API_KEY="your_openai_key_here"
export ANTHROPIC_API_KEY="your_anthropic_key_here"

# Add to ~/.bashrc for persistence
echo 'export OPENAI_API_KEY="your_openai_key_here"' >> ~/.bashrc
echo 'export ANTHROPIC_API_KEY="your_anthropic_key_here"' >> ~/.bashrc

# Verify keys are set
echo "OpenAI key status: ${OPENAI_API_KEY:+set}"
echo "Anthropic key status: ${ANTHROPIC_API_KEY:+set}"

# Test API connectivity
python -c '
import os
import openai
client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Test connection"}],
    max_tokens=5
)
print("OpenAI API test successful")
'
```

1. Create New Idea JSON File
```bash
# Create ideas directory
mkdir -p templates/nanoGPT_lite/ideas/

# Create new idea file
cat > templates/nanoGPT_lite/ideas/new_idea.json << 'EOL'
{
    "idea": "Using AI to optimize renewable energy sources",
    "Name": "renewable_energy_optimization",
    "Title": "AI-Driven Optimization of Renewable Energy Sources",
    "Experiment": "Implement an AI model to analyze and optimize renewable energy resource allocation",
    "Interestingness": 8,
    "Feasibility": 7,
    "Novelty": 6,
    "novel": true
}
EOL
```
1.1. Verify JSON Format and Structure
```bash
# Install jq for JSON validation if not installed
sudo apt-get install jq -y

# Verify JSON syntax and format
jq '.' templates/nanoGPT_lite/ideas/new_idea.json || {
    echo "✗ Invalid JSON syntax"
    exit 1
}

# Check required fields and validate structure
jq -e '
    if (has("idea") and has("Name") and has("Title") and
        has("Experiment") and has("Interestingness") and
        has("Feasibility") and has("Novelty") and has("novel")) then
        "✓ All required fields present"
    else
        "✗ Missing required fields" | halt_error
    end
' templates/nanoGPT_lite/ideas/new_idea.json

# Verify field types and value ranges
jq -e '
    if ((.Name | type) != "string") then "✗ Name must be string" | halt_error
    elif ((.Title | type) != "string") then "✗ Title must be string" | halt_error
    elif ((.Experiment | type) != "string") then "✗ Experiment must be string" | halt_error
    elif ((.Interestingness | type) != "number") then "✗ Interestingness must be number" | halt_error
    elif ((.Feasibility | type) != "number") then "✗ Feasibility must be number" | halt_error
    elif ((.Novelty | type) != "number") then "✗ Novelty must be number" | halt_error
    elif ((.novel | type) != "boolean") then "✗ novel must be boolean" | halt_error
    elif (.Interestingness < 1 or .Interestingness > 10) then "✗ Interestingness must be 1-10" | halt_error
    elif (.Feasibility < 1 or .Feasibility > 10) then "✗ Feasibility must be 1-10" | halt_error
    elif (.Novelty < 1 or .Novelty > 10) then "✗ Novelty must be 1-10" | halt_error
    else "✓ All fields have correct types and ranges"
    end
' templates/nanoGPT_lite/ideas/new_idea.json

# Print success message if all checks pass
echo "✓ JSON verification complete - all checks passed"
```
2. Modify launch_scientist.py
- Open the launch_scientist.py file
- Ensure your idea is included in the AVAILABLE_IDEAS list
- Verify the model configuration matches your requirements

Example modification:
```python
# Add your idea to the available ideas
AVAILABLE_IDEAS = [
    "renewable_energy_optimization",
    # ... existing ideas ...
]

# Configure model parameters if needed
model_config = {
    "n_layer": 8,
    "n_head": 8,
    "n_embd": 512,
    # Add any custom parameters
}
```

3. Run the Experiment

3.1. Environment Verification
```bash
# Navigate to project directory
cd AI-Scientist

# Verify Python environment
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import openai; print(f'OpenAI SDK version: {openai.__version__}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

# Verify CUDA availability (if using GPU)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

3.2. Basic Test Configuration
```bash
# Run basic test with default settings
python launch_scientist.py \
    --idea templates/nanoGPT_lite/ideas/new_idea.json \
    --model gpt-4o-2024-05-13 \
    --device cpu \
    --batch-size 32

# Monitor training progress in real-time
tail -f templates/nanoGPT_lite/run_*/train.log

# Check experiment status
ls -l templates/nanoGPT_lite/run_*/
```

3.3. Advanced Configuration (Optional)
```bash
# Run with custom parameters
python launch_scientist.py \
    --idea templates/nanoGPT_lite/ideas/new_idea.json \
    --model gpt-4o-2024-05-13 \
    --num-runs 3 \
    --device cpu \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --max-iters 1000
```

3.4. Troubleshooting
- If import errors occur, verify all dependencies are installed
- For CUDA errors, ensure GPU drivers are properly configured
- Check train.log for detailed error messages
- Verify JSON file path and format if idea loading fails

4. Monitor Results

4.1. Real-time Progress Monitoring
```bash
# Watch training progress in real-time
tail -f templates/nanoGPT_lite/run_*/train.log

# Monitor GPU usage (if using GPU)
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv -l 1

# Check process status
ps aux | grep launch_scientist.py
```

4.2. Analyze Performance Metrics in final_info.json
```bash
# View complete final metrics
cat templates/nanoGPT_lite/run_*/final_info.json

# Extract and format key performance metrics
jq -r '
    "=== Performance Metrics ===\n" +
    "Training Time: \(.training_time)s\n" +
    "Inference Speed: \(.inference_speed) tokens/s\n" +
    "Final Training Loss: \(.training_loss)\n" +
    "Final Validation Loss: \(.validation_loss)\n" +
    "Memory Usage: \(.memory_usage // "N/A") MB\n" +
    "Total Parameters: \(.total_parameters // "N/A")\n" +
    "=== Training History ===\n" +
    "Loss Improvement: \((.training_loss_history[0] - .training_loss_history[-1]) // "N/A")\n" +
    "Convergence Rate: \(.convergence_rate // "N/A")\n" +
    "=== Resource Utilization ===\n" +
    "Peak Memory: \(.peak_memory // "N/A") MB\n" +
    "Average GPU Utilization: \(.avg_gpu_utilization // "N/A")%"
' templates/nanoGPT_lite/run_*/final_info.json

# Compare with baseline performance
echo "=== Performance vs Baseline ==="
jq -r --argfile baseline templates/nanoGPT_lite/run_0/final_info.json '
    .training_time as $time |
    .inference_speed as $speed |
    .training_loss as $loss |
    $baseline |
    {
        "Time Difference (s)": ($time - .training_time),
        "Speed Improvement (%)": (($speed - .inference_speed) / .inference_speed * 100),
        "Loss Improvement (%)": ((.training_loss - $loss) / .training_loss * 100)
    } | to_entries[] | "\(.key): \(.value)"
' templates/nanoGPT_lite/run_*/final_info.json

# Generate performance report
cat > analyze_performance.py << 'EOL'
import json
import sys
from pathlib import Path

def analyze_performance(run_dir):
    with open(Path(run_dir) / 'final_info.json') as f:
        data = json.load(f)

    # Calculate key performance indicators
    loss_history = data.get('training_loss_history', [])
    convergence_speed = len(loss_history) / data.get('training_time', 1)
    stability = calculate_stability(loss_history)

    print(f"\nPerformance Analysis for {Path(run_dir).name}")
    print("=" * 50)
    print(f"Model Efficiency:")
    print(f"- Training Speed: {data.get('training_time', 'N/A')} seconds")
    print(f"- Inference Speed: {data.get('inference_speed', 'N/A')} tokens/s")
    print(f"- Convergence Speed: {convergence_speed:.2f} iterations/s")

    print("\nModel Quality:")
    print(f"- Final Training Loss: {data.get('training_loss', 'N/A')}")
    print(f"- Final Validation Loss: {data.get('validation_loss', 'N/A')}")
    print(f"- Training Stability: {stability:.2%}")

    print("\nResource Usage:")
    print(f"- Peak Memory: {data.get('peak_memory', 'N/A')} MB")
    print(f"- GPU Utilization: {data.get('avg_gpu_utilization', 'N/A')}%")

def calculate_stability(loss_history):
    if not loss_history:
        return 0
    variations = [abs(b - a) for a, b in zip(loss_history, loss_history[1:])]
    return 1 - (sum(variations) / len(variations)) / max(loss_history)

if __name__ == '__main__':
    for run_dir in Path('templates/nanoGPT_lite').glob('run_*'):
        analyze_performance(run_dir)
EOL

# Run performance analysis
python analyze_performance.py
```

4.3. Visualize Training Progress
```python
# Create visualization script
cat > plot_metrics.py << 'EOL'
import json
import matplotlib.pyplot as plt
import glob
import os

def plot_metrics(run_dir):
    with open(os.path.join(run_dir, 'final_info.json')) as f:
        data = json.load(f)

    # Plot training progress
    plt.figure(figsize=(10, 6))
    plt.plot(data.get('training_loss_history', []), label='Training Loss')
    plt.plot(data.get('validation_loss_history', []), label='Validation Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Training Progress - {os.path.basename(run_dir)}')
    plt.legend()
    plt.savefig(os.path.join(run_dir, 'training_progress.png'))
    plt.close()

# Plot for all runs
for run_dir in glob.glob('templates/nanoGPT_lite/run_*'):
    plot_metrics(run_dir)
EOL

# Run visualization
python plot_metrics.py
```

4.4. Performance Comparison
Baseline Performance (Run 0):
- Training time: 243.34s
- Inference speed: 89.88 tokens/s
- Training loss: 2.574
- Validation loss: 2.574

Key Metrics to Monitor:
- Training and validation loss curves
- Training time and throughput
- Memory usage and GPU utilization
- Model convergence speed
- Final model performance

Important Notes:
- Monitor training loss for signs of overfitting
- Compare metrics with baseline run
- Document any anomalies or interesting patterns
- Save visualizations for documentation
- Track resource utilization for optimization

4.6. Save Configuration for Reproducibility
```bash
# Create experiment configuration backup
mkdir -p experiment_backups

# Save complete experiment configuration
cat > experiment_backups/experiment_config.json << 'EOL'
{
    "experiment_name": "renewable_energy_optimization",
    "timestamp": "$(date -u +"%Y-%m-%d_%H-%M-%S")",
    "model_config": {
        "model": "gpt-4o-2024-05-13",
        "batch_size": 32,
        "learning_rate": 1e-4,
        "max_iters": 1000,
        "device": "cpu"
    },
    "idea_config": $(cat templates/nanoGPT_lite/ideas/new_idea.json),
    "environment": {
        "python_version": "$(python --version | cut -d' ' -f2)",
        "torch_version": "$(python -c 'import torch; print(torch.__version__)')",
        "cuda_available": "$(python -c 'import torch; print(torch.cuda.is_available())')",
        "system_info": "$(uname -a)"
    }
}
EOL

# Save complete experiment results
cp -r templates/nanoGPT_lite/run_* experiment_backups/

# Create experiment archive
tar -czf "experiment_backups/experiment_$(date -u +"%Y-%m-%d_%H-%M-%S").tar.gz" \
    experiment_backups/experiment_config.json \
    experiment_backups/run_* \
    templates/nanoGPT_lite/ideas/new_idea.json

# Verify backup
echo "Experiment configuration and results backed up successfully:"
ls -lh experiment_backups/
```
