from __future__ import annotations

import ast
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from urllib import error, request

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


app = FastAPI(
    title="Multi-Agent AI Decision System",
    description="Hackathon-ready API that simulates CEO, Finance, Marketing, and Operations agents.",
    version="1.0.0",
)
FRONTEND_FILE = Path(__file__).resolve().parents[1] / "frontend" / "index.html"


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_load_env_file(PROJECT_ROOT / ".env")
_load_env_file(PROJECT_ROOT / "backend" / ".env")

LLM_API_KEY = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
LLM_MODEL = os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
LLM_BASE_URL = os.getenv("LLM_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "250"))
LLM_USE_RESPONSE_FORMAT = os.getenv("LLM_USE_RESPONSE_FORMAT", "false").strip().lower() in {"1", "true", "yes", "on"}
REQUIRE_LLM = os.getenv("REQUIRE_LLM", "false").strip().lower() in {"1", "true", "yes", "on"}


class ProblemInput(BaseModel):
    company_type: str = Field(..., examples=["E-commerce Startup"])
    core_problem: str = Field(..., examples=["Low customer retention and declining repeat purchases"])
    budget: str = Field(..., examples=["₹5 lakh"])
    timeline: str = Field(..., examples=["3 months"])
    constraints: str = Field(..., examples=["Small team (5 members), limited marketing budget"])


class AgentDiscussion(BaseModel):
    agent: str
    thoughts: str
    decision: str


class DecisionOutput(BaseModel):
    problem: str
    agents_discussion: List[AgentDiscussion]
    conflicts_identified: str
    resolution: str
    final_decision: str
    execution_plan: List[str]


@dataclass
class AgentState:
    thoughts: str
    decision: str


def _extract_json_block(text: str) -> Dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    # First try direct parse.
    try:
        return json.loads(text)
    except ValueError:
        pass

    # Fallback: find the first JSON object block in mixed text.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("No JSON object found in model output.")


def _extract_content_text(body: Dict) -> str:
    message = body["choices"][0]["message"]
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                chunks.append(part["text"])
        return "\n".join(chunks).strip()
    return str(content)


def _chat_completion(payload: Dict, headers: Dict[str, str]) -> Dict:
    req = request.Request(
        f"{LLM_BASE_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with request.urlopen(req, timeout=45) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def _coerce_decision_output(parsed: Dict, data: ProblemInput) -> Dict:
    # Make parser resilient across weaker/free models that may miss fields or vary casing.
    problem = _clean_display_text(parsed.get("problem") or _problem_statement(data))
    conflicts = _clean_display_text(parsed.get("conflicts_identified") or parsed.get("conflicts") or "No major strategic conflicts detected.")
    resolution = _clean_display_text(parsed.get("resolution") or "Proceed with aligned strategy and weekly metric reviews.")
    final_decision = _format_final_decision(parsed.get("final_decision") or parsed.get("decision") or "Proceed with a practical phased plan.")

    raw_agents = parsed.get("agents_discussion") or parsed.get("agent_discussion") or []
    agents_discussion: List[Dict[str, str]] = []
    if isinstance(raw_agents, list):
        for item in raw_agents:
            if isinstance(item, dict):
                agent_name = _clean_display_text(item.get("agent") or "Agent")
                thoughts = _clean_display_text(item.get("thoughts") or item.get("analysis") or "")
                decision = _clean_display_text(item.get("decision") or item.get("recommendation") or "")
                agents_discussion.append({"agent": agent_name, "thoughts": thoughts, "decision": decision})
    if not agents_discussion:
        agents_discussion = [
            {"agent": "CEO", "thoughts": "Strategic direction focused on business impact.", "decision": final_decision},
            {"agent": "Finance", "thoughts": "Budget and ROI constraints considered.", "decision": "Prioritize measurable, high-ROI actions."},
            {"agent": "Marketing", "thoughts": "Customer behavior and acquisition/retention levers analyzed.", "decision": "Run segmented campaigns and optimize funnel."},
            {"agent": "Operations", "thoughts": "Execution capacity and timelines assessed.", "decision": "Use phased rollout with clear ownership and KPIs."},
        ]

    exec_plan = parsed.get("execution_plan") or parsed.get("plan") or []
    if isinstance(exec_plan, str):
        exec_plan = [line.strip("- ").strip() for line in exec_plan.splitlines() if line.strip()]
    if not isinstance(exec_plan, list):
        exec_plan = []
    exec_plan = [_clean_display_text(step) for step in exec_plan if _clean_display_text(step)]
    if not exec_plan:
        exec_plan = [
            "Week 1: Define baseline KPIs and customer segments.",
            "Week 2-4: Launch high-impact initiatives and track metrics weekly.",
            "Week 5-8: Optimize based on measured results and remove low-performing actions.",
            "Week 9-12: Scale proven levers and finalize next-quarter roadmap.",
        ]

    return {
        "problem": problem,
        "agents_discussion": agents_discussion,
        "conflicts_identified": conflicts,
        "resolution": resolution,
        "final_decision": final_decision,
        "execution_plan": exec_plan,
    }


def _clean_display_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{") or text.startswith("["):
            parsed_text = _parse_structured_text(text)
            if parsed_text is not None:
                return _clean_display_text(parsed_text)
            return " ".join(text.split())
        return " ".join(text.split())
    if isinstance(value, list):
        return "; ".join(_clean_display_text(item) for item in value if _clean_display_text(item))
    if isinstance(value, dict):
        priority_keys = [
            "summary",
            "recommendation",
            "decision",
            "final_decision",
            "resolution",
            "action",
            "rationale",
        ]
        parts: List[str] = []
        for key in priority_keys:
            if value.get(key):
                label = key.replace("_", " ").title()
                parts.append(f"{label}: {_clean_display_text(value[key])}")
        if parts:
            return " ".join(parts)
        return " ".join(f"{key.replace('_', ' ').title()}: {_clean_display_text(item)}" for key, item in value.items())
    return " ".join(str(value).split())


def _format_final_decision(value) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{") or text.startswith("["):
            parsed_text = _parse_structured_text(text)
            if parsed_text is not None:
                return _format_final_decision(parsed_text)
            return _clean_display_text(text)
        return _clean_display_text(text)
    if isinstance(value, list):
        return " ".join(_clean_display_text(item) for item in value if _clean_display_text(item))
    if isinstance(value, dict):
        lines: List[str] = []
        summary = value.get("summary") or value.get("final_decision") or value.get("decision") or value.get("recommendation")
        rationale = value.get("rationale") or value.get("reason") or value.get("why")
        actions = value.get("actions") or value.get("recommended_actions") or value.get("next_steps")
        budget = value.get("budget_allocation") or value.get("budget")

        if summary:
            lines.append(_clean_display_text(summary))
        if rationale:
            lines.append(f"Reason: {_clean_display_text(rationale)}")
        if budget:
            lines.append(f"Budget focus: {_clean_display_text(budget)}")
        if actions:
            lines.append(f"Recommended actions: {_clean_display_text(actions)}")

        return " ".join(lines) if lines else _clean_display_text(value)
    return _clean_display_text(value)


def _parse_structured_text(text: str):
    try:
        return json.loads(text)
    except ValueError:
        pass

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None

    if isinstance(parsed, (dict, list, str, int, float, bool)) or parsed is None:
        return parsed
    return None


def _decision_from_unstructured_text(text: str, data: ProblemInput) -> Dict:
    clean = " ".join((text or "").split())
    if not clean:
        raise ValueError("Empty LLM text response.")

    snippet = clean[:700]
    return {
        "problem": _problem_statement(data),
        "agents_discussion": [
            {
                "agent": "CEO",
                "thoughts": f"Strategy synthesized from live LLM output: {snippet[:220]}",
                "decision": "Prioritize high-impact initiatives aligned with timeline and constraints.",
            },
            {
                "agent": "Finance",
                "thoughts": "Budget and efficiency signals were inferred from the generated output.",
                "decision": "Use ROI-based spend controls and protect contribution margin.",
            },
            {
                "agent": "Marketing",
                "thoughts": "Customer lifecycle and reactivation opportunities were identified from the response.",
                "decision": "Run segmented retention and win-back campaigns with targeted incentives.",
            },
            {
                "agent": "Operations",
                "thoughts": "Execution feasibility was mapped to small-team constraints.",
                "decision": "Deploy in weekly sprints with KPI checkpoints and clear owners.",
            },
        ],
        "conflicts_identified": "Potential trade-off between growth speed and budget discipline.",
        "resolution": "Prioritize measurable experiments first; scale only validated levers.",
        "final_decision": snippet,
        "execution_plan": [
            "Week 1: Set baseline metrics and segment customers.",
            "Week 2-4: Launch retention and win-back initiatives.",
            "Week 5-8: Optimize channels based on ROI and engagement.",
            "Week 9-12: Scale winning playbooks and document SOPs.",
        ],
    }


def _llm_multi_agent_decision(data: ProblemInput) -> Tuple[DecisionOutput | None, str | None]:
    if not LLM_API_KEY:
        return None, "LLM key missing. Set LLM_API_KEY or OPENAI_API_KEY."

    system_prompt = (
        "You are an enterprise multi-agent decision engine. "
        "Return only valid JSON. No markdown. No extra text."
    )
    user_prompt = f"""
Solve this business problem with 4 independent agents (CEO, Finance, Marketing, Operations).
Agents may agree/disagree, review each other, and resolve conflicts logically.
Be realistic and practical.

Input:
- company_type: {data.company_type}
- core_problem: {data.core_problem}
- budget: {data.budget}
- timeline: {data.timeline}
- constraints: {data.constraints}

Return EXACTLY this schema:
{{
  "problem": "<user input>",
  "agents_discussion": [
    {{"agent":"CEO","thoughts":"...","decision":"..."}},
    {{"agent":"Finance","thoughts":"...","decision":"..."}},
    {{"agent":"Marketing","thoughts":"...","decision":"..."}},
    {{"agent":"Operations","thoughts":"...","decision":"..."}}
  ],
  "conflicts_identified":"...",
  "resolution":"...",
  "final_decision":"...",
  "execution_plan":["Step 1...","Step 2..."]
}}
"""
    payload = {
        "model": LLM_MODEL,
        "temperature": 0.4,
        "max_tokens": LLM_MAX_TOKENS,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if LLM_USE_RESPONSE_FORMAT:
        payload["response_format"] = {"type": "json_object"}
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    # Optional headers some OpenAI-compatible providers (e.g. OpenRouter) accept.
    llm_http_referer = os.getenv("LLM_HTTP_REFERER", "").strip()
    llm_app_title = os.getenv("LLM_APP_TITLE", "").strip()
    if llm_http_referer:
        headers["HTTP-Referer"] = llm_http_referer
    if llm_app_title:
        headers["X-Title"] = llm_app_title

    try:
        body = _chat_completion(payload, headers)
        content = _extract_content_text(body)
        try:
            parsed = _extract_json_block(content)
        except ValueError:
            # Recovery pass: ask model to convert its own text into strict JSON schema.
            repair_payload = {
                "model": LLM_MODEL,
                "temperature": 0.0,
                "max_tokens": min(LLM_MAX_TOKENS, 300),
                "messages": [
                    {"role": "system", "content": "Return only valid JSON. No markdown."},
                    {
                        "role": "user",
                        "content": (
                            "Convert the following text into JSON with keys: "
                            "problem, agents_discussion[{agent,thoughts,decision}], conflicts_identified, "
                            "resolution, final_decision, execution_plan[]\n\n"
                            f"TEXT:\n{content[:3500]}"
                        ),
                    },
                ],
            }
            if LLM_USE_RESPONSE_FORMAT:
                repair_payload["response_format"] = {"type": "json_object"}
            repair_body = _chat_completion(repair_payload, headers)
            repair_text = _extract_content_text(repair_body)
            parsed = _extract_json_block(repair_text)

        normalized = _coerce_decision_output(parsed, data)
        return DecisionOutput.model_validate(normalized), None
    except error.HTTPError as exc:
        detail = ""
        try:
            detail_raw = exc.read().decode("utf-8", errors="ignore")
            if detail_raw:
                try:
                    detail_obj = json.loads(detail_raw)
                    # Common OpenAI-compatible error shapes.
                    detail = (
                        detail_obj.get("error", {}).get("message")
                        or detail_obj.get("message")
                        or detail_raw
                    )
                except ValueError:
                    detail = detail_raw
        except Exception:
            detail = ""
        detail = detail.strip().replace("\n", " ")
        if detail:
            return None, f"LLM HTTP error: {exc.code} - {detail[:220]}"
        return None, f"LLM HTTP error: {exc.code}"
    except error.URLError:
        return None, "LLM network error."
    except (KeyError, ValueError):
        # Final safeguard: accept plain-text LLM output and shape it into required schema.
        try:
            body = _chat_completion(payload, headers)
            content = _extract_content_text(body)
            synthesized = _decision_from_unstructured_text(content, data)
            return DecisionOutput.model_validate(synthesized), None
        except Exception:
            return None, "LLM returned invalid JSON payload."


def _parse_budget_inr_lakh(budget: str) -> float:
    lower = budget.lower().strip()
    numeric = ""
    for ch in lower:
        if ch.isdigit() or ch == ".":
            numeric += ch
    if not numeric:
        return 0.0

    value = float(numeric)
    if "crore" in lower:
        return value * 100.0
    if "lakh" in lower or "lac" in lower:
        return value
    # Assume raw rupee value if no unit is present.
    if value > 100000:
        return value / 100000.0
    return value


def _identify_focus(core_problem: str) -> Dict[str, bool]:
    p = core_problem.lower()
    return {
        "retention": any(k in p for k in ["retention", "repeat", "churn", "inactive"]),
        "acquisition": any(k in p for k in ["acquisition", "new users", "traffic", "lead"]),
        "conversion": any(k in p for k in ["conversion", "checkout", "cart abandon"]),
        "revenue_drop": any(k in p for k in ["declining", "revenue down", "sales down"]),
    }


def _initial_agent_views(data: ProblemInput) -> Dict[str, AgentState]:
    budget_lakh = _parse_budget_inr_lakh(data.budget)
    focus = _identify_focus(data.core_problem)
    small_budget = budget_lakh <= 10
    short_timeline = "month" in data.timeline.lower() and any(x in data.timeline for x in ["1", "2", "3"])
    small_team = any(k in data.constraints.lower() for k in ["small team", "5", "lean", "limited team"])

    ceo_priority = "retention" if focus["retention"] else "balanced growth"
    ceo_state = AgentState(
        thoughts=(
            f"The company is a {data.company_type} facing '{data.core_problem}'. "
            f"Given budget ({data.budget}) and timeline ({data.timeline}), strategy must prioritize "
            f"{ceo_priority} with measurable 90-day outcomes."
        ),
        decision=(
            "Set top-level target: improve repeat purchase behavior and stabilize revenue quality. "
            "Run a phased 12-week plan: diagnose, launch retention levers, scale winners."
        ),
    )

    finance_state = AgentState(
        thoughts=(
            "Financial discipline is critical. We should avoid broad discounts that erode margin and "
            "allocate spend only to measurable channels."
        ),
        decision=(
            "Allocate budget across CRM automation, targeted incentives, remarketing, lightweight UX fixes, "
            "and contingency. Review weekly ROI and cut non-performing activities quickly."
        ),
    )

    if small_budget:
        finance_state.decision += " Keep ad spend capped and performance-based due to limited capital."

    marketing_state = AgentState(
        thoughts=(
            "Customer lifecycle appears weak. We need segmented campaigns for post-purchase nurture, "
            "win-back, and second-order activation."
        ),
        decision=(
            "Launch lifecycle tracks via email/WhatsApp + retargeting. Focus on second-purchase offers, "
            "personalized recommendations, and referral/loyalty mechanics."
        ),
    )

    operations_state = AgentState(
        thoughts=(
            "Execution must match team capacity. Prefer low-code tools, clear ownership, and weekly sprints "
            "instead of heavy custom builds."
        ),
        decision=(
            "Define a simple operating cadence: one owner per stream, KPI dashboard, SOPs for campaign launch, "
            "and fast feedback loop between support and growth."
        ),
    )

    if short_timeline or small_team:
        operations_state.decision += " Avoid complex engineering; use plugins and automation first."

    return {
        "CEO": ceo_state,
        "Finance": finance_state,
        "Marketing": marketing_state,
        "Operations": operations_state,
    }


def _discussion_and_refinement(states: Dict[str, AgentState], data: ProblemInput) -> Tuple[str, str]:
    conflicts: List[str] = []
    resolutions: List[str] = []
    budget_lakh = _parse_budget_inr_lakh(data.budget)
    low_budget = budget_lakh <= 10

    # Conflict 1: Marketing vs Finance on discount/ad aggressiveness.
    if low_budget:
        conflicts.append(
            "Marketing requested stronger remarketing and incentive intensity, while Finance warned about margin pressure."
        )
        states["Marketing"].decision = (
            states["Marketing"].decision
            + " Prioritize segmented incentives over blanket discounts and focus on high-intent audiences."
        )
        resolutions.append(
            "Use targeted offers with spend caps and strict ROI thresholds instead of broad discounting."
        )

    # Conflict 2: CEO ambition vs Operations capacity.
    if "small team" in data.constraints.lower() or "5" in data.constraints:
        conflicts.append(
            "CEO pushed for fast transformation; Operations flagged implementation risk with a 5-member team."
        )
        states["CEO"].decision = (
            states["CEO"].decision
            + " Scope changes to high-impact, low-complexity initiatives in first 6 weeks."
        )
        resolutions.append(
            "Sequence work in weekly sprints and implement only low-complexity levers first."
        )

    conflict_text = "; ".join(conflicts) if conflicts else "No major strategic conflicts detected."
    resolution_text = (
        " ".join(resolutions)
        if resolutions
        else "Proceed with aligned strategy and weekly metric reviews."
    )
    return conflict_text, resolution_text


def _build_execution_plan(data: ProblemInput) -> List[str]:
    return [
        "Week 1: Build baseline metrics (repeat purchase rate, 30/60-day retention, CAC payback, contribution margin).",
        "Week 1-2: Segment customers into new, repeat, dormant, and high-value cohorts.",
        "Week 2-3: Configure CRM automation for post-purchase journeys and second-order nudges.",
        "Week 3-4: Launch win-back flows for inactive users with personalized product recommendations.",
        "Week 4-6: Introduce referral/loyalty pilot targeting high-value and recent repeat cohorts.",
        "Week 5-8: Run controlled remarketing campaigns for high-intent audiences with spend caps.",
        "Week 7-10: Deploy lightweight UX fixes (checkout friction, reorder discovery, trust boosters).",
        "Week 10-12: Scale top-performing levers, stop weak channels, and finalize next-quarter retention roadmap.",
    ]


def _problem_statement(data: ProblemInput) -> str:
    return (
        f"{data.company_type}: {data.core_problem}. "
        f"Budget: {data.budget}. Timeline: {data.timeline}. Constraints: {data.constraints}."
    )


def run_multi_agent_decision(data: ProblemInput) -> Tuple[DecisionOutput, str, str | None]:
    llm_result, llm_error = _llm_multi_agent_decision(data)
    if llm_result:
        return llm_result, "llm", None

    if REQUIRE_LLM:
        raise ValueError(llm_error or "LLM output required, but generation failed.")

    states = _initial_agent_views(data)
    conflicts, resolution = _discussion_and_refinement(states, data)

    final_decision = (
        "Execute a retention-first 90-day program combining lifecycle automation, targeted win-back, "
        "and operationally simple execution. Scale only initiatives that increase repeat purchases "
        "without harming contribution margin."
    )

    discussions = [
        AgentDiscussion(agent="CEO", thoughts=states["CEO"].thoughts, decision=states["CEO"].decision),
        AgentDiscussion(agent="Finance", thoughts=states["Finance"].thoughts, decision=states["Finance"].decision),
        AgentDiscussion(agent="Marketing", thoughts=states["Marketing"].thoughts, decision=states["Marketing"].decision),
        AgentDiscussion(agent="Operations", thoughts=states["Operations"].thoughts, decision=states["Operations"].decision),
    ]

    return DecisionOutput(
        problem=_problem_statement(data),
        agents_discussion=discussions,
        conflicts_identified=conflicts,
        resolution=resolution,
        final_decision=final_decision,
        execution_plan=_build_execution_plan(data),
    ), "fallback", llm_error


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> Dict[str, str]:
    return {
        "message": "Multi-Agent AI Decision System is running.",
        "mode": "llm+fallback",
        "app": "/app",
        "docs": "/docs",
        "health": "/health",
        "source_header": "X-Decision-Source",
        "llm_required": str(REQUIRE_LLM).lower(),
        "decision_endpoint": "/decision (POST)",
    }


@app.get("/app")
def app_ui() -> FileResponse:
    return FileResponse(FRONTEND_FILE)


@app.get("/mode")
def mode() -> Dict[str, str]:
    return {
        "llm_base_url": LLM_BASE_URL,
        "llm_model": LLM_MODEL,
        "llm_configured": str(bool(LLM_API_KEY)).lower(),
        "llm_required": str(REQUIRE_LLM).lower(),
        "decision_source_header": "X-Decision-Source",
    }


@app.post("/decision", response_model=DecisionOutput)
def decision(data: ProblemInput, response: Response) -> DecisionOutput:
    try:
        result, source, reason = run_multi_agent_decision(data)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    response.headers["X-Decision-Source"] = source
    if reason:
        response.headers["X-Decision-Fallback-Reason"] = reason[:180]
    return result
