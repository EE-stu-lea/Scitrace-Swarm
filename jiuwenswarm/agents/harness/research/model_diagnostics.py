from __future__ import annotations

import asyncio
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import os
from pathlib import Path
import socket
import ssl
from typing import Any
from urllib.parse import urlparse

from jiuwenswarm.common.config import get_config, get_config_raw, get_default_models
from jiuwenswarm.common.utils import get_config_file
from jiuwenswarm.symphony.llm import LLMConfig, extract_message_content


def _package_version(name: str) -> str | None:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


def _configured_secrets() -> set[str]:
    secrets = {
        str(os.getenv("API_KEY") or "").strip(),
        str(os.getenv("OPENAI_API_KEY") or "").strip(),
        str(os.getenv("DASHSCOPE_API_KEY") or "").strip(),
        str(os.getenv("ANTHROPIC_API_KEY") or "").strip(),
    }
    return {item for item in secrets if item}


def sanitize_text(value: Any) -> str:
    text = str(value or "")
    for secret in _configured_secrets():
        text = text.replace(secret, "[REDACTED]")
    return text


def _exception_chain(exc: BaseException | None) -> list[dict[str, Any]]:
    chain: list[dict[str, Any]] = []
    seen: set[int] = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(
            {
                "type": type(current).__name__,
                "message": sanitize_text(current),
                "status_code": getattr(current, "status_code", None),
            }
        )
        current = current.__cause__ or current.__context__
    return chain


def _result_ok(payload: dict[str, Any]) -> bool:
    return bool(payload.get("success"))


def _field_source(raw_entry: dict[str, Any] | None, field: str, env_name: str) -> str:
    if raw_entry is None:
        return "jiuwenswarm_defaults"
    raw_client = raw_entry.get("model_client_config") if isinstance(raw_entry, dict) else None
    raw_value = ""
    if isinstance(raw_client, dict):
        raw_value = str(raw_client.get(field) or "")
    env_value = str(os.getenv(env_name) or "").strip()
    if "${" in raw_value:
        return "environment" if env_value else "jiuwenswarm_defaults"
    if raw_value.strip():
        return "yaml"
    if env_value:
        return "environment"
    return "jiuwenswarm_defaults"


def _select_raw_default_entry(raw_config: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    models = raw_config.get("models") if isinstance(raw_config, dict) else None
    if not isinstance(models, dict):
        return "none", None
    defaults = models.get("defaults")
    if isinstance(defaults, list) and defaults:
        for entry in defaults:
            if isinstance(entry, dict) and entry.get("is_default") is True:
                return "models.defaults", entry
        first = defaults[0]
        return "models.defaults", first if isinstance(first, dict) else None
    default_entry = models.get("default")
    if isinstance(default_entry, dict):
        return "models.default", default_entry
    return "none", None


def resolve_model_config_diagnostics() -> dict[str, Any]:
    raw_config = get_config_raw() or {}
    resolved_config = get_config() or {}
    selected_source, raw_entry = _select_raw_default_entry(raw_config)
    try:
        defaults = get_default_models(resolved_config)
    except Exception as exc:  # pragma: no cover - defensive
        defaults = []
        defaults_error = exc
    else:
        defaults_error = None
    selected_entry = next((item for item in defaults if item.get("is_default") is True), defaults[0] if defaults else None)
    has_declared_default = selected_source != "none"

    llm_config_error: Exception | None = None
    llm_config: LLMConfig | None = None
    try:
        llm_config = LLMConfig.from_default_model()
    except Exception as exc:
        llm_config_error = exc

    client_config = {}
    model_name = ""
    if llm_config is not None:
        client_config = dict(llm_config.model_client_config or {})
        model_name = llm_config.model
    elif isinstance(selected_entry, dict):
        client_config = dict(selected_entry.get("model_client_config") or {})
        model_name = str(
            (selected_entry.get("model_config_obj") or {}).get("model")
            or client_config.get("model_name")
            or ""
        ).strip()

    api_base = str(client_config.get("api_base") or "").strip().rstrip("/")
    provider = str(client_config.get("client_provider") or "").strip()
    configured = {
        "provider": provider,
        "model_name": model_name,
        "api_base": api_base,
        "api_key_configured": bool(str(client_config.get("api_key") or "").strip()),
    }
    sources = {
        "provider": _field_source(raw_entry, "client_provider", "MODEL_PROVIDER"),
        "model_name": _field_source(raw_entry, "model_name", "MODEL_NAME"),
        "api_base": _field_source(raw_entry, "api_base", "API_BASE"),
        "api_key": _field_source(raw_entry, "api_key", "API_KEY"),
    }
    selected_entry_source = (
        "environment"
        if has_declared_default and raw_entry is not None and all(value == "environment" for value in sources.values())
        else selected_source
    )
    return {
        "config_file": str(get_config_file()),
        "selected_entry_source": selected_entry_source,
        "has_declared_default": has_declared_default,
        "resolved": configured,
        "sources": sources,
        "env_presence": {
            "API_BASE": bool(str(os.getenv("API_BASE") or "").strip()),
            "API_KEY": bool(str(os.getenv("API_KEY") or "").strip()),
            "MODEL_NAME": bool(str(os.getenv("MODEL_NAME") or "").strip()),
            "MODEL_PROVIDER": bool(str(os.getenv("MODEL_PROVIDER") or "").strip()),
        },
        "timeout_seconds": float(client_config.get("timeout") or 0.0),
        "verify_ssl": bool(client_config.get("verify_ssl", True)),
        "custom_headers_configured": bool(client_config.get("custom_headers")),
        "llm_config_error": _exception_chain(llm_config_error) if llm_config_error else [],
        "defaults_error": _exception_chain(defaults_error) if defaults_error else [],
        "python_version": sanitize_text(os.sys.version),
        "sdk_versions": {
            "openai": _package_version("openai"),
            "anthropic": _package_version("anthropic"),
            "openjiuwen": _package_version("openjiuwen"),
            "jiuwenswarm": _package_version("workswarm"),
        },
    }


def _api_base_parts(api_base: str) -> dict[str, Any]:
    parsed = urlparse(api_base)
    scheme = parsed.scheme or "https"
    host = parsed.hostname or ""
    port = parsed.port or (443 if scheme == "https" else 80)
    return {
        "scheme": scheme,
        "host": host,
        "port": port,
        "path": parsed.path or "",
    }


def collect_proxy_info(api_base: str) -> dict[str, Any]:
    from openjiuwen.core.common.security.url_utils import UrlUtils

    env_keys = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy")
    present = {key: bool(str(os.getenv(key) or "").strip()) for key in env_keys}
    selected = UrlUtils.get_global_proxy_url(api_base) if api_base else None
    return {
        "detected_proxy_env": present,
        "selected_proxy": sanitize_text(selected) if selected else "",
    }


def collect_dns_tcp_tls(api_base: str, *, verify_ssl: bool, timeout_seconds: float) -> dict[str, Any]:
    parts = _api_base_parts(api_base)
    host = parts["host"]
    port = parts["port"]
    diagnostics = {
        "api_base_parts": parts,
        "dns": {"success": False, "addresses": [], "error_chain": []},
        "tcp": {"success": False, "error_chain": []},
        "tls": {"success": False, "version": "", "cipher": "", "error_chain": []},
    }
    if not host:
        return diagnostics
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        addresses = sorted({item[4][0] for item in infos if item[4]})
        diagnostics["dns"] = {"success": True, "addresses": addresses, "error_chain": []}
    except Exception as exc:
        diagnostics["dns"] = {"success": False, "addresses": [], "error_chain": _exception_chain(exc)}
        return diagnostics

    sock = None
    try:
        sock = socket.create_connection((host, port), timeout=max(1.0, timeout_seconds))
        diagnostics["tcp"] = {"success": True, "error_chain": []}
        if parts["scheme"] == "https":
            context = ssl.create_default_context() if verify_ssl else ssl._create_unverified_context()
            with context.wrap_socket(sock, server_hostname=host) as tls_socket:
                cipher = tls_socket.cipher() or ("", "", 0)
                diagnostics["tls"] = {
                    "success": True,
                    "version": tls_socket.version() or "",
                    "cipher": cipher[0] or "",
                    "error_chain": [],
                }
            sock = None
    except Exception as exc:
        target = "tls" if parts["scheme"] == "https" and diagnostics["tcp"]["success"] else "tcp"
        diagnostics[target] = {
            "success": False,
            "version": "",
            "cipher": "",
            "error_chain": _exception_chain(exc),
        } if target == "tls" else {"success": False, "error_chain": _exception_chain(exc)}
    finally:
        if sock is not None:
            sock.close()
    return diagnostics


def _openai_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Return exactly pong."},
        {"role": "user", "content": prompt},
    ]


def _anthropic_messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def _extract_openai_content(response: Any) -> str:
    try:
        choice = response.choices[0]
        return sanitize_text(choice.message.content)
    except Exception:
        return sanitize_text(response)


def _extract_anthropic_content(response: Any) -> str:
    content = []
    for block in getattr(response, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            content.append(text)
    return sanitize_text("\n".join(content))


async def _run_openai_level1_async(client_config: dict[str, Any], model_name: str, timeout_seconds: float) -> dict[str, Any]:
    import httpx
    from openai import AsyncOpenAI
    from openjiuwen.core.common.security.url_utils import UrlUtils
    from openjiuwen.core.common.utils.header_utils import sanitize_headers
    from openjiuwen.core.foundation.llm import ModelClientConfig
    from openjiuwen.core.foundation.llm.model_clients.openai_model_client import _normalize_openai_base_url
    from openjiuwen.core.foundation.llm.utils.endpoint_profiles import normalize_model_client_config

    normalized = normalize_model_client_config(ModelClientConfig(**client_config))
    http_client = httpx.AsyncClient(
        proxy=UrlUtils.get_global_proxy_url(normalized.api_base),
        verify=bool(normalized.verify_ssl),
        limits=httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
            keepalive_expiry=60.0,
        ),
    )
    kwargs = {
        "api_key": normalized.api_key,
        "base_url": _normalize_openai_base_url(normalized.api_base),
        "http_client": http_client,
        "timeout": max(1.0, timeout_seconds),
        "max_retries": 0,
    }
    headers = sanitize_headers(getattr(normalized, "custom_headers", None))
    if headers:
        kwargs["default_headers"] = headers
    client = AsyncOpenAI(**kwargs)
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=_openai_messages("ping"),
            temperature=0.0,
            max_tokens=8,
            timeout=max(1.0, timeout_seconds),
        )
        return {"success": True, "content": _extract_openai_content(response), "error_chain": []}
    finally:
        await client.close()


def _run_openai_level1_sync(client_config: dict[str, Any], model_name: str, timeout_seconds: float) -> dict[str, Any]:
    import httpx
    from openai import OpenAI
    from openjiuwen.core.common.security.url_utils import UrlUtils
    from openjiuwen.core.common.utils.header_utils import sanitize_headers
    from openjiuwen.core.foundation.llm import ModelClientConfig
    from openjiuwen.core.foundation.llm.model_clients.openai_model_client import _normalize_openai_base_url
    from openjiuwen.core.foundation.llm.utils.endpoint_profiles import normalize_model_client_config

    normalized = normalize_model_client_config(ModelClientConfig(**client_config))
    http_client = httpx.Client(
        proxy=UrlUtils.get_global_proxy_url(normalized.api_base),
        verify=bool(normalized.verify_ssl),
        limits=httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
            keepalive_expiry=60.0,
        ),
    )
    kwargs = {
        "api_key": normalized.api_key,
        "base_url": _normalize_openai_base_url(normalized.api_base),
        "http_client": http_client,
        "timeout": max(1.0, timeout_seconds),
        "max_retries": 0,
    }
    headers = sanitize_headers(getattr(normalized, "custom_headers", None))
    if headers:
        kwargs["default_headers"] = headers
    client = OpenAI(**kwargs)
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=_openai_messages("ping"),
            temperature=0.0,
            max_tokens=8,
            timeout=max(1.0, timeout_seconds),
        )
        return {"success": True, "content": _extract_openai_content(response), "error_chain": []}
    finally:
        client.close()


async def _run_anthropic_level1_async(client_config: dict[str, Any], model_name: str, timeout_seconds: float) -> dict[str, Any]:
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(
        api_key=client_config.get("api_key"),
        base_url=client_config.get("api_base"),
        timeout=max(1.0, timeout_seconds),
    )
    response = await client.messages.create(
        model=model_name,
        system="Return exactly pong.",
        messages=_anthropic_messages("ping"),
        max_tokens=8,
        temperature=0.0,
    )
    return {"success": True, "content": _extract_anthropic_content(response), "error_chain": []}


def _run_anthropic_level1_sync(client_config: dict[str, Any], model_name: str, timeout_seconds: float) -> dict[str, Any]:
    from anthropic import Anthropic

    client = Anthropic(
        api_key=client_config.get("api_key"),
        base_url=client_config.get("api_base"),
        timeout=max(1.0, timeout_seconds),
    )
    response = client.messages.create(
        model=model_name,
        system="Return exactly pong.",
        messages=_anthropic_messages("ping"),
        max_tokens=8,
        temperature=0.0,
    )
    return {"success": True, "content": _extract_anthropic_content(response), "error_chain": []}


def _call_result(fn, *args) -> dict[str, Any]:
    try:
        return fn(*args)
    except Exception as exc:
        return {"success": False, "content": "", "error_chain": _exception_chain(exc)}


def run_level1_sdk_probe(client_config: dict[str, Any], model_name: str, *, timeout_seconds: float) -> dict[str, Any]:
    provider = str(client_config.get("client_provider") or "").strip()
    openai_compatible = {"OpenAI", "OpenRouter", "SiliconFlow", "DashScope", "DeepSeek", "InferenceAffinity", "AscendAffinity"}
    if provider == "Anthropic":
        sync_result = _call_result(_run_anthropic_level1_sync, client_config, model_name, timeout_seconds)
        async_result = _call_result(lambda cfg, model, timeout: asyncio.run(_run_anthropic_level1_async(cfg, model, timeout)), client_config, model_name, timeout_seconds)
    elif provider in openai_compatible:
        sync_result = _call_result(_run_openai_level1_sync, client_config, model_name, timeout_seconds)
        async_result = _call_result(lambda cfg, model, timeout: asyncio.run(_run_openai_level1_async(cfg, model, timeout)), client_config, model_name, timeout_seconds)
    else:
        sync_result = {"success": False, "content": "", "error_chain": [{"type": "UnsupportedProvider", "message": f"unsupported Level 1 provider: {provider}", "status_code": None}]}
        async_result = {"success": False, "content": "", "error_chain": [{"type": "UnsupportedProvider", "message": f"unsupported Level 1 provider: {provider}", "status_code": None}]}
    return {
        "provider": provider,
        "sync_client": sync_result,
        "async_client": async_result,
    }


def run_level2_jiuwenswarm_probe(*, timeout_seconds: float) -> dict[str, Any]:
    try:
        llm_config = LLMConfig.from_default_model()
        model = llm_config.create_model()
    except Exception as exc:
        return {"success": False, "content": "", "error_chain": _exception_chain(exc)}

    async def _invoke():
        response = await model.invoke(
            messages=_openai_messages("ping"),
            temperature=0.0,
            top_p=1.0,
            timeout=max(1.0, timeout_seconds),
        )
        return extract_message_content(response)

    try:
        content = asyncio.run(_invoke())
        return {"success": True, "content": sanitize_text(content), "error_chain": []}
    except Exception as exc:
        return {"success": False, "content": "", "error_chain": _exception_chain(exc)}


def _suggestions(summary: dict[str, Any]) -> list[str]:
    suggestions: list[str] = []
    resolved = summary["resolved_model"]
    provider = str(resolved["provider"] or "")
    api_base = str(resolved["api_base"] or "")
    parts = summary["network"]["api_base_parts"]
    proxy = summary["proxy"]["selected_proxy"]
    level1 = summary["level1_sdk_call"]
    level2 = summary["level2_jiuwenswarm_call"]

    if not api_base:
        suggestions.append("API Base URL 为空，请先确认环境变量或 JiuwenSwarm 默认模型配置。")
    if provider == "OpenAI" and "anthropic" in api_base.lower():
        suggestions.append("当前 provider 与 API Base 看起来不匹配：OpenAI 路径却指向 Anthropic 端点。")
    path = str(parts.get("path") or "")
    scheme = str(parts.get("scheme") or "https")
    if provider == "Anthropic" and "/v1" in path:
        suggestions.append("Anthropic provider 通常不应直接使用 OpenAI 风格的 /v1 基础路径。")
    if provider == "OpenAI" and api_base and "/v1" not in path and "localhost" not in api_base.lower():
        suggestions.append("当前 API Base 可能缺少 /v1；请确认服务是否要求 OpenAI 兼容路径。")
    if summary["network"]["dns"]["success"] is False:
        suggestions.append("DNS 解析失败，请检查 API Base、VPN、代理或企业 DNS 策略。")
    if summary["network"]["tcp"]["success"] is False:
        suggestions.append("TCP 连接失败，请检查代理、VPN、防火墙或目标服务连通性。")
    if scheme == "https" and summary["network"]["tls"]["success"] is False:
        suggestions.append("TLS 握手失败，请检查证书链、代理拦截或 verify_ssl 配置。")
    if proxy:
        suggestions.append("检测到代理配置，建议核对 HTTP_PROXY/HTTPS_PROXY/NO_PROXY 是否命中正确地址。")
    if _result_ok(level1["sync_client"]) and not _result_ok(level1["async_client"]):
        suggestions.append("同步 SDK 调用成功但异步失败，优先排查 Async 客户端、事件循环或代理连接池。")
    if _result_ok(level1["async_client"]) and not _result_ok(level2):
        suggestions.append("Level 1 成功而 Level 2 失败，问题更可能在 JiuwenSwarm 配置解析、provider 归一化或模型构造链。")
    if summary["config_resolution"]["sources"]["api_key"] == "environment" and not summary["config_resolution"]["env_presence"]["API_KEY"]:
        suggestions.append("诊断显示 API Key 依赖环境变量，但当前进程中未检测到 API_KEY。")
    if not suggestions:
        suggestions.append("未发现明显的静态配置错误；请结合 Level 1/Level 2 调用结果继续定位服务端可达性或账号权限。")
    return suggestions


def _sanitize_call_result(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(payload or {})
    cleaned["content"] = sanitize_text(cleaned.get("content", ""))
    chain = []
    for item in cleaned.get("error_chain", []) or []:
        if isinstance(item, dict):
            chain.append(
                {
                    "type": item.get("type"),
                    "message": sanitize_text(item.get("message", "")),
                    "status_code": item.get("status_code"),
                }
            )
    cleaned["error_chain"] = chain
    return cleaned


def diagnose_model_connection() -> dict[str, Any]:
    resolution = resolve_model_config_diagnostics()
    resolved_model = resolution["resolved"]
    api_base = resolved_model["api_base"]
    # A full Team invocation may allow many minutes, but a connection probe
    # should fail fast before any costly research stage is started.
    timeout_seconds = min(30.0, max(5.0, float(resolution["timeout_seconds"] or 30.0)))
    network = collect_dns_tcp_tls(
        api_base,
        verify_ssl=bool(resolution["verify_ssl"]),
        timeout_seconds=min(timeout_seconds, 15.0),
    )
    proxy = collect_proxy_info(api_base)
    level1 = {"provider": resolved_model["provider"], "sync_client": {"success": False, "content": "", "error_chain": []}, "async_client": {"success": False, "content": "", "error_chain": []}}
    client_config = None
    try:
        client_config = dict((LLMConfig.from_default_model().model_client_config or {}))
    except Exception:
        try:
            defaults = get_default_models(get_config() or {})
        except Exception:
            defaults = []
        entry = next((item for item in defaults if item.get("is_default") is True), defaults[0] if defaults else None)
        if isinstance(entry, dict):
            client_config = dict(entry.get("model_client_config") or {})
    if resolved_model["provider"] and resolved_model["model_name"] and resolved_model["api_base"] and resolved_model["api_key_configured"]:
        client_config = client_config or {}
        client_config["client_provider"] = client_config.get("client_provider") or resolved_model["provider"]
        client_config["api_base"] = client_config.get("api_base") or resolved_model["api_base"]
        client_config["timeout"] = client_config.get("timeout") or timeout_seconds
        client_config["verify_ssl"] = client_config.get("verify_ssl", bool(resolution["verify_ssl"]))
        level1 = run_level1_sdk_probe(
            client_config,
            resolved_model["model_name"],
            timeout_seconds=timeout_seconds,
        )
    level1 = {
        "provider": sanitize_text(level1.get("provider", "")),
        "sync_client": _sanitize_call_result(level1.get("sync_client") or {}),
        "async_client": _sanitize_call_result(level1.get("async_client") or {}),
    }
    if level1["sync_client"].get("success") and level1["async_client"].get("success"):
        level2 = _sanitize_call_result(run_level2_jiuwenswarm_probe(timeout_seconds=timeout_seconds))
    else:
        level2 = {
            "success": False,
            "content": "",
            "error_chain": [{"type": "SkippedProbe", "message": "Skipped JiuwenSwarm probe because an SDK probe failed.", "status_code": None}],
        }
    summary = {
        "config_resolution": resolution,
        "resolved_model": {
            "provider": resolved_model["provider"],
            "model_name": resolved_model["model_name"],
            "api_base": resolved_model["api_base"],
            "api_key": {"configured": resolved_model["api_key_configured"]},
        },
        "proxy": proxy,
        "network": network,
        "level1_sdk_call": level1,
        "level2_jiuwenswarm_call": level2,
    }
    summary["suggestions"] = _suggestions(summary)
    return summary


def write_model_diagnostics(path: str | Path) -> dict[str, Any]:
    payload = diagnose_model_connection()
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return payload


__all__ = [
    "diagnose_model_connection",
    "resolve_model_config_diagnostics",
    "sanitize_text",
    "write_model_diagnostics",
]
