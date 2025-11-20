# Use the official vLLM OpenAI-compatible image
# You can pin to a specific version instead of :latest if you like.
FROM vllm/vllm-openai:latest

ENV PYTHONUNBUFFERED=1

# Workdir for any extra scripts / code you add
WORKDIR /app

# Copy repo content (entrypoint, any benchmark helpers, etc.)
COPY . /app

ENV TORCH_CUDA_ARCH_LIST=12.1a
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Make sure the "bench" extras are available.
# --no-deps so we DON'T try to reinstall torch / vllm and break CUDA.
RUN pip install --no-cache-dir "vllm[bench]" --no-deps

# Our entrypoint chooses between:
#   - vllm serve        (MODE=serve, the default)
#   - vllm serve + vllm bench serve (MODE=bench-serve)
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["serve"]
