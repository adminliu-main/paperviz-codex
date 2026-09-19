# PaperVizAgent

PaperVizAgent turns a paper's method section and figure caption into academic methodology diagrams. It uses a Planner → Visualizer → Critic loop: the model first plans the figure, creates an image, inspects it, and iterates on the result.

This fork adds a practical local mode: it can reuse a user's logged-in Codex CLI for reasoning **and native image generation**, so a Gemini or OpenAI API key is not required for the main diagram workflow.

## Highlights

- Generate multiple scientific-diagram candidates from a method section and caption.
- Review and refine diagrams through multi-round visual critique.
- Choose either cloud APIs/gateways or a locally logged-in Codex CLI backend.
- Use Codex's built-in raster image-generation tool; generated PNGs are retained locally.
- English/Chinese switcher in the Streamlit Demo.
- Works on macOS, Linux, and Windows.

## Quick start: Codex mode (recommended)

Codex mode requires an installed Codex CLI and a ChatGPT/Codex login. It uses your Codex plan allowance rather than a separate provider API key.

### macOS / Linux

```bash
uv venv --python 3.12
source .venv/bin/activate  # fish: source .venv/bin/activate.fish
uv pip install -r requirements.txt

codex login
cp configs/model_config.template.yaml configs/model_config.yaml
```

### Windows PowerShell

```powershell
uv venv --python 3.12
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt

codex login
Copy-Item configs\model_config.template.yaml configs\model_config.yaml
```

Edit `configs/model_config.yaml`:

```yaml
defaults:
  backend: "codex"
  model_name: "codex"
  image_model_name: "codex-image"

codex:
  command: "codex"
  model: ""          # Empty = use the Codex CLI default model
  timeout_seconds: 300
```

Start the app:

```bash
uv run streamlit run demo.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`. The language selector is at the top of the left sidebar.

For a first run, use one candidate and one critic round. Each candidate can create an initial image plus one image for every successful critic round.

## Where Codex images are stored

Every native Codex image is preserved at:

```text
outputs/codex/<unique-run-id>/diagram.png
```

The Demo displays only the final successful image for each candidate. The intermediate images stay in `outputs/codex/` for inspection. This directory is ignored by Git.

## Cloud API and gateway mode

Set `backend` to `api` to use the original provider SDKs:

```yaml
defaults:
  backend: "api"
  model_name: "your-gemini-text-model"
  image_model_name: "gpt-image-2"

api_keys:
  google_api_key: "your-gemini-key"
  openai_api_key: "your-openai-key"

endpoints:
  gemini_base_url: "https://your-gemini-gateway.example/v1beta"
  openai_base_url: "https://your-openai-gateway.example/v1"
```

Environment variables override YAML values:

```bash
export GEMINI_BASE_URL="https://your-gemini-gateway.example/v1beta"
export GOOGLE_API_KEY="your-gemini-key"
export OPENAI_BASE_URL="https://your-openai-gateway.example/v1"
export OPENAI_API_KEY="your-openai-key"
```

Gateway requirements:

- The Gemini endpoint must support Gemini's `generateContent` protocol.
- The OpenAI endpoint must support `/v1/images/generations` and return `b64_json` for `gpt-image-*` generation.
- A chat-only OpenAI gateway cannot create images.

## Using the Demo

1. Choose a pipeline mode:
   - `demo_planner_critic`: Planner → Visualizer → Critic → Visualizer.
   - `demo_full`: Retriever → Planner → Stylist → Visualizer → Critic → Visualizer.
2. Paste the paper's method section and the desired figure caption.
3. Choose the number of candidates, aspect ratio, and critic rounds.
4. Select **Generate Candidates**.
5. View, download, or inspect the evolution timeline for each result.

If the PaperBananaBench reference data is not present, retrieval automatically falls back to no-reference mode; generation can still continue without few-shot examples.

The **Refine Image** tab currently uses the original Gemini/Vertex image-editing route. The Codex backend covers the candidate-generation pipeline.

## Command-line batch mode

Place an input split under `data/PaperBananaBench/` and run:

```bash
python main.py \
  --task_name diagram \
  --split_name test \
  --exp_mode dev_planner_critic \
  --retrieval_setting none \
  --max_critic_rounds 1
```

| Mode | Pipeline |
| --- | --- |
| `vanilla` | Direct rendering only |
| `dev_planner` | Retriever → Planner → Visualizer |
| `dev_planner_stylist` | Retriever → Planner → Stylist → Visualizer |
| `dev_planner_critic` | Planner → Visualizer → Critic loop |
| `dev_full` | Full retrieval, styling, and critique pipeline |
| `demo_planner_critic` | Demo pipeline without dataset evaluation |
| `demo_full` | Full demo pipeline without dataset evaluation |

## Troubleshooting

### The page keeps generating

The app waits for every candidate and critic round to finish before it displays the final grid. Check active Codex jobs and generated files:

```bash
ps -ax -o pid=,etime=,command= | rg 'codex exec'
find outputs/codex -name diagram.png | wc -l
```

On Windows, use Task Manager to inspect `codex.exe`/`node.exe`, or run:

```powershell
Get-Process | Where-Object { $_.ProcessName -match 'codex|node' }
```

### Stop a run

In the terminal running Streamlit, press `Ctrl+C`. Do not use `killall python`: it can close unrelated Python programs.

### Gateway image 404

If the log says `/images/generations` returned 404, confirm that `openai_base_url` is the API root such as `https://gateway.example/v1`, not the full image route and not a chat-only endpoint.

### Record image API diagnostics

For cloud API mode only:

```bash
PAPERVIZ_DEBUG_LOG=1 uv run streamlit run demo.py
```

Failures are appended to `logs/openai_image_debug.jsonl`. Keys and image base64 payloads are intentionally excluded.

## Security and cost

- `configs/model_config.yaml` is Git-ignored. Never commit API keys.
- Codex mode does not make API-key calls, but consumes your Codex plan allowance.
- API/gateway mode is billed by the selected provider or relay.
- Codex image outputs and Demo result files may contain paper content; treat them as local research artifacts.

## License and upstream

This project is based on [Google Research PaperVizAgent](https://github.com/google-research/papervizagent) and retains the repository's Apache-2.0 license.
