Local LLM setup (text-generation-webui + GPT4All - quick guide)

This guide shows how to run a free, small local LLM on Windows using WSL (recommended) or directly on Windows with GPT4All.

Option A (recommended): WSL + text-generation-webui + ggml GPT4All model (CPU)

1. Install WSL and Ubuntu (from Microsoft Store) if not already installed.

2. Open WSL terminal (Ubuntu) and run:

```bash
# clone webui
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Create models directory and download a small GPT4All ggml model
mkdir -p models
# Example: download ggml-gpt4all-j.bin (replace with specific model link)
# For example, from GPT4All releases or mirror; download manually into models/
# cp /mnt/c/Users/you/Downloads/ggml-gpt4all-j.bin models/

# Start the server with API enabled on port 5000
python server.py --model models/ggml-gpt4all-j.bin --api --listen --port 5000
```

3. The local API will be available at http://127.0.0.1:5000/api/v1/generate (text-generation-webui versions may use slightly different endpoints; check server output).

Option B: GPT4All native (Windows)

- Download GPT4All Desktop or GPT4All server and a small model. Follow GPT4All docs to start a local HTTP API.

Notes & tips
- Models are several GB. Ensure enough disk space.
- CPU inference is slow; consider a small model (1-3B) for interactive use.
- Keep the server running and do not expose it to the public internet.
- If you prefer not to use WSL, run webui on a Linux host and point backend LOCAL_LLM_API to it.

Testing the server
- Use curl or Postman to POST a small JSON to the generate endpoint and confirm it returns generated text.

Example curl (adjust endpoint shape to match your webui version):

```bash
curl -s -X POST "http://127.0.0.1:5000/api/v1/generate" -H "Content-Type: application/json" -d '{"prompt":"Hello world","max_new_tokens":16}'
```

If you want, I can add a small PowerShell script to automate starting WSL and launching the server (requires correct model path).
