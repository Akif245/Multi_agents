# Multi-Agent AI Decision System (Hackathon)

This project provides a working FastAPI backend that simulates 4 business agents:

- CEO
- Finance
- Marketing
- Operations

It generates structured multi-agent discussion, conflict resolution, and a final execution plan.

## 1) Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2) Run API

```powershell
# Real AI output (recommended):
$env:OPENAI_API_KEY="your_openai_api_key"

# Optional OpenAI-compatible override (other provider):
# $env:LLM_API_KEY="your_provider_key"
# $env:LLM_BASE_URL="https://your-provider.com/v1"
# $env:LLM_MODEL="model-name"

# Force real output only (no fallback):
# $env:REQUIRE_LLM="true"

uvicorn backend.main:app --reload
```

API starts at:

- `http://127.0.0.1:8000`
- `http://127.0.0.1:8000/app` (Hackathon UI)
- Docs: `http://127.0.0.1:8000/docs`

## 3) Test Endpoint

`POST /decision`

Example body:

```json
{
  "company_type": "E-commerce Startup",
  "core_problem": "Low customer retention and declining repeat purchases",
  "budget": "₹5 lakh",
  "timeline": "3 months",
  "constraints": "Small team (5 members), limited marketing budget"
}
```

The response matches this format:

- `problem`
- `agents_discussion` (CEO/Finance/Marketing/Operations)
- `conflicts_identified`
- `resolution`
- `final_decision`
- `execution_plan`

## Real vs Fallback Output

- If `OPENAI_API_KEY` is set, `/decision` uses a real LLM call for dynamic output.
- If key is missing or API call fails, it automatically falls back to local rule-based logic.
- If `REQUIRE_LLM=true`, the API returns `503` instead of fallback when LLM fails.
- Decision source is exposed in response header: `X-Decision-Source` (`llm` or `fallback`).
