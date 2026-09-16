from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ResourceAuditSnapshot:
    session_id: str
    role: str = ""
    member_name: str = ""
    started_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    invoke_count: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    tool_failures: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    retries: int = 0
    total_runtime_seconds: float = 0.0
    total_tool_seconds: float = 0.0
    total_experiment_seconds: float = 0.0
    tools_by_name: dict[str, int] = field(default_factory=dict)
    tool_seconds_by_name: dict[str, float] = field(default_factory=dict)
    model_calls_by_name: dict[str, int] = field(default_factory=dict)
    model_tokens_by_name: dict[str, int] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResourceReporter:
    """Write machine-readable and markdown resource reports."""

    def __init__(
        self,
        output_root: str | Path,
        *,
        json_filename: str = "resource_report.json",
        markdown_filename: str = "resource_report.md",
        nest_by_session: bool = True,
    ) -> None:
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.json_filename = json_filename
        self.markdown_filename = markdown_filename
        self.nest_by_session = bool(nest_by_session)

    def session_dir(self, session_id: str) -> Path:
        if not self.nest_by_session:
            return self.output_root
        directory = self.output_root / session_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def write_snapshot(self, snapshot: ResourceAuditSnapshot) -> tuple[Path, Path]:
        directory = self.session_dir(snapshot.session_id)
        snapshot.updated_at = utc_now_iso()
        json_path = directory / self.json_filename
        markdown_path = directory / self.markdown_filename
        json_path.write_text(
            json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        markdown_path.write_text(self.render_markdown(snapshot), encoding="utf-8")
        return json_path, markdown_path

    @staticmethod
    def render_markdown(snapshot: ResourceAuditSnapshot) -> str:
        lines = [
            "# Resource Report",
            "",
            f"- session_id: `{snapshot.session_id}`",
            f"- role: `{snapshot.role or '-'}`",
            f"- member_name: `{snapshot.member_name or '-'}`",
            f"- invoke_count: `{snapshot.invoke_count}`",
            f"- model_calls: `{snapshot.model_calls}`",
            f"- tool_calls: `{snapshot.tool_calls}`",
            f"- tool_failures: `{snapshot.tool_failures}`",
            f"- input_tokens: `{snapshot.input_tokens}`",
            f"- output_tokens: `{snapshot.output_tokens}`",
            f"- total_tokens: `{snapshot.total_tokens}`",
            f"- retries: `{snapshot.retries}`",
            f"- total_runtime_seconds: `{snapshot.total_runtime_seconds:.3f}`",
            f"- total_tool_seconds: `{snapshot.total_tool_seconds:.3f}`",
            f"- total_experiment_seconds: `{snapshot.total_experiment_seconds:.3f}`",
            "",
            "## Tool Usage",
        ]
        if snapshot.tools_by_name:
            for tool_name in sorted(snapshot.tools_by_name):
                count = snapshot.tools_by_name[tool_name]
                seconds = snapshot.tool_seconds_by_name.get(tool_name, 0.0)
                lines.append(f"- `{tool_name}`: count=`{count}`, seconds=`{seconds:.3f}`")
        else:
            lines.append("- none")

        lines.extend(["", "## Model Usage"])
        if snapshot.model_calls_by_name:
            for model_name in sorted(snapshot.model_calls_by_name):
                count = snapshot.model_calls_by_name[model_name]
                tokens = snapshot.model_tokens_by_name.get(model_name, 0)
                lines.append(f"- `{model_name}`: calls=`{count}`, tokens=`{tokens}`")
        else:
            lines.append("- none")

        if snapshot.extra:
            lines.extend(["", "## Extra"])
            for key in sorted(snapshot.extra):
                lines.append(f"- `{key}`: `{snapshot.extra[key]}`")

        lines.append("")
        return "\n".join(lines)
