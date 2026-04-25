For this Project, building a local AI assistant using a Small Language Model (SLM), the presenter breaks the process down into three distinct phases to demonstrate engineering maturity regarding privacy, latency, and hardware constraints (5:26-8:50):

Phase 1: Setup and Performance Benchmarking (6:11-6:51)

Tools: Install Ollama to run models locally.
Model Selection: Choose an open-source model in the 3B to 7B parameter range (e.g., Llama 3.2 3B, Phi-4, or Mistral 7B).
Development: Build either a simple command-line tool or a FastAPI wrapper.
Benchmarking: Rigorously measure and document inference performance, including tokens per second, time to first token, and total response latency.
Phase 2: Structure, Determinism, and Control (6:52-7:43)

Output Handling: Enforce a JSON output schema on model responses.
Validation: Use Pydantic to validate outputs and implement a retry mechanism that catches invalid responses and re-prompts before failing.
Experimentation: Test the model at different temperature settings (e.g., 0 vs. 0.7) and document the variance to show an understanding of the stochastic nature of language models.
Phase 3: Model Comparison Study (7:44-8:50)

Benchmarking: Compare three different models on the same hardware using a standardized set of 30-50 test prompts.
Metrics: Document memory usage, tokens per second, and output quality.
Report: Compile the findings into a concise technical report. If you want to stand out, try using quantized versions of the models (e.g., GGUF Q4 or Q5) and document the specific quality-vs-speed trade-offs involved.