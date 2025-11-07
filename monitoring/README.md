## Monitoring

Use Docker Compose to start both services (Prometheus and Grafana):

```
docker compose up -d
```

In Grafana (e.g., http://localhost:3000) add a Prometheus data source with the URL:

```
http://prometheus:9090
```

Import dashboards using grafana.json

Add a panel for Prefill Tokens/Second vs Decode Tokens/Second using:

```
# Prefill Tokens/Second:
rate(vllm:request_prompt_tokens_sum[30s])/rate(vllm:request_prefill_time_seconds_sum[30s])

# Decode Tokens/Second:
rate(vllm:request_generation_tokens_sum[30s])/rate(vllm:request_decode_time_seconds_sum[30s])

```

