"""SPEC-2026-002 Task 6 — the fail-closed ``pre_memory_load`` gate.

Agent initialization freezes a system-prompt snapshot of ``MEMORY.md`` /
``USER.md`` at load time (``tools/memory_tool.py`` ``load_from_disk``), and the
whole memory block used to live inside a ``try`` whose tail is
``except Exception: pass  # Memory is optional``. Anything dispatched inside
that ``try`` is swallowed, so it cannot be a gate.

Task 6 adds one cross-surface lifecycle seam that runs AFTER plugin discovery
and BEFORE ``load_from_disk()``, dispatched OUTSIDE that ``try``:

* no registered hook            -> current behavior, unchanged
* hook allows                   -> normal init
* hook blocks / raises / hangs  -> initialization aborts loudly
* genuine ``MemoryStore`` load faults stay swallowable upstream, but fail loud
  when the profile marks the gate required (``memory.pre_memory_load_required``)
"""

import ast
import inspect
import threading
from pathlib import Path

import pytest
import yaml

import hermes_cli.plugins as plugins_mod
from hermes_cli.plugins import PluginManager
from run_agent import AIAgent

HOOK = "pre_memory_load"


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class _FakeOpenAI:
    def __init__(self, **kw):
        self.api_key = kw.get("api_key", "test")
        self.base_url = kw.get("base_url", "http://test")

    def close(self):
        pass


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """A throwaway profile home — never touch a real HERMES_HOME."""
    home = tmp_path / ".hermes"
    (home / "memories").mkdir(parents=True)
    (home / "memories" / "MEMORY.md").write_text(
        "task06-memory-line\n", encoding="utf-8"
    )
    (home / "memories" / "USER.md").write_text(
        "task06-user-line\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _write_memory_config(home: Path, **memory_section) -> None:
    (home / "config.yaml").write_text(
        yaml.safe_dump({"memory": memory_section}), encoding="utf-8"
    )


@pytest.fixture
def register_gate(monkeypatch):
    """Install ``pre_memory_load`` callbacks on an isolated plugin manager."""
    saved_by_home = dict(plugins_mod._plugin_managers_by_home)
    saved_pointer = plugins_mod._plugin_manager

    def _register(*callbacks):
        mgr = PluginManager()
        # Skip real discovery: the temp home has no plugins and discovery
        # would clear the hooks we just wired.
        mgr._discovered = True
        mgr._hooks[HOOK] = list(callbacks)
        monkeypatch.setattr(plugins_mod, "_plugin_manager", mgr)
        return mgr

    yield _register

    plugins_mod._plugin_managers_by_home.clear()
    plugins_mod._plugin_managers_by_home.update(saved_by_home)
    plugins_mod._plugin_manager = saved_pointer


def _make_agent(monkeypatch, *, platform="cli", skip_memory=False,
                enabled_toolsets=None, disabled_toolsets=None):
    monkeypatch.setattr("run_agent.get_tool_definitions", lambda **kw: [])
    monkeypatch.setattr("run_agent.check_toolset_requirements", lambda: {})
    monkeypatch.setattr("run_agent.OpenAI", _FakeOpenAI)
    return AIAgent(
        api_key="test-key",
        base_url="http://test",
        provider="openrouter",
        api_mode="chat_completions",
        max_iterations=1,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=skip_memory,
        platform=platform,
        enabled_toolsets=enabled_toolsets,
        disabled_toolsets=disabled_toolsets,
    )


# ---------------------------------------------------------------------------
# 1. Registration / policy placement
# ---------------------------------------------------------------------------

def test_hook_is_a_valid_hook():
    assert HOOK in plugins_mod.VALID_HOOKS


def test_hook_is_in_the_fail_closed_policy_set():
    """Alongside pre_tool_call — NOT among the bounded fail-open hooks."""
    assert HOOK in plugins_mod._HOOK_TIMEOUT_FAIL_CLOSED_HOOKS
    assert "pre_tool_call" in plugins_mod._HOOK_TIMEOUT_FAIL_CLOSED_HOOKS
    assert HOOK not in plugins_mod._HOOK_TIMEOUT_BOUNDED_HOOKS


def test_hook_callback_errors_fail_closed():
    """A raised callback must become a block directive, not a warning log."""
    assert HOOK in plugins_mod._HOOK_ERROR_FAIL_CLOSED_HOOKS


def test_shell_hooks_cannot_register_the_gate():
    """Shell hooks have no channel for this directive — refuse loudly."""
    assert HOOK in plugins_mod.SHELL_UNSUPPORTED_HOOKS


def test_raising_callback_yields_block_directive_from_invoke_hook():
    mgr = PluginManager()
    mgr._discovered = True

    def boom(**_kwargs):
        raise RuntimeError("gate exploded")

    mgr._hooks[HOOK] = [boom]
    results = mgr.invoke_hook(HOOK)
    assert results and results[0].get("action") == "block"
    assert "gate exploded" in results[0].get("message", "")


# ---------------------------------------------------------------------------
# 2. Structural guard: the dispatch is OUTSIDE the memory try/except
# ---------------------------------------------------------------------------

def _gate_call_nodes(tree):
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_enforce_pre_memory_load_gate"
    ]


def _memory_load_try_nodes(tree):
    """The ``try`` blocks that wrap ``MemoryStore.load_from_disk()``."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for stmt in node.body:
            for inner in ast.walk(stmt):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Attribute)
                    and inner.func.attr == "load_from_disk"
                ):
                    found.append(node)
                    break
            else:
                continue
            break
    return found


def _swallowing_try_nodes(tree):
    """``try`` blocks whose handler can swallow (contains a bare ``pass``)."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        and any(
            isinstance(stmt, ast.Pass)
            for handler in node.handlers
            for stmt in handler.body
        )
    ]


def test_gate_dispatch_is_outside_the_memory_try_block():
    """Delta 1 regression guard.

    If the dispatch were moved back inside the ``except Exception: pass``
    memory try block, a raising gate would be swallowed. Assert structurally
    that the dispatch is not nested in the load_from_disk try, nor in any
    other try whose handler can swallow.
    """
    source = Path("agent/agent_init.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    gate_calls = _gate_call_nodes(tree)
    assert gate_calls, "expected an enforce_pre_memory_load_gate() dispatch"

    memory_tries = _memory_load_try_nodes(tree)
    assert memory_tries, "expected the MemoryStore.load_from_disk() try block"

    def _calls_inside(try_nodes):
        return {
            id(call)
            for try_node in try_nodes
            for stmt in try_node.body
            for call in _gate_call_nodes(stmt)
        }

    inside_memory_try = _calls_inside(memory_tries)
    assert not any(id(call) in inside_memory_try for call in gate_calls), (
        "pre_memory_load dispatch must NOT sit inside the "
        "'except Exception: pass  # Memory is optional' try block"
    )

    inside_swallowing = _calls_inside(_swallowing_try_nodes(tree))
    assert not any(id(call) in inside_swallowing for call in gate_calls), (
        "pre_memory_load dispatch must not sit inside any swallowing try"
    )


def test_gate_runs_after_plugin_discovery_and_before_disk_load(
    monkeypatch, hermes_home, register_gate
):
    order = []

    real_discover = plugins_mod.discover_plugins

    def _discover(*a, **kw):
        order.append("discover")
        return real_discover(*a, **kw)

    monkeypatch.setattr(plugins_mod, "discover_plugins", _discover)

    from tools.memory_tool import MemoryStore

    real_load = MemoryStore.load_from_disk

    def _load(self, *a, **kw):
        order.append("load_from_disk")
        return real_load(self, *a, **kw)

    monkeypatch.setattr(MemoryStore, "load_from_disk", _load)

    register_gate(lambda **kw: order.append("gate") or {"action": "allow"})

    _make_agent(monkeypatch)

    assert "gate" in order and "load_from_disk" in order
    assert order.index("discover") < order.index("gate") < order.index("load_from_disk")


# ---------------------------------------------------------------------------
# 3. Allow paths
# ---------------------------------------------------------------------------

def test_no_registered_hook_loads_memory_unchanged(monkeypatch, hermes_home):
    agent = _make_agent(monkeypatch)
    assert agent._memory_store is not None
    assert "task06-memory-line" in agent._memory_store.format_for_system_prompt("memory")


def test_well_behaved_hook_allows_normal_init(monkeypatch, hermes_home, register_gate):
    register_gate(lambda **kw: {"action": "allow"})
    agent = _make_agent(monkeypatch)
    assert agent._memory_store is not None
    assert "task06-memory-line" in agent._memory_store.format_for_system_prompt("memory")


def test_hook_returning_none_is_allow(monkeypatch, hermes_home, register_gate):
    register_gate(lambda **kw: None)
    agent = _make_agent(monkeypatch)
    assert agent._memory_store is not None


def test_payload_carries_profile_scoped_context(monkeypatch, hermes_home, register_gate):
    _write_memory_config(hermes_home, projection_source="runtime/memory/MEMORY.md")
    seen = {}

    def _gate(**kwargs):
        seen.update(kwargs)
        return {"action": "allow"}

    register_gate(_gate)
    agent = _make_agent(monkeypatch, platform="telegram")

    assert seen["hermes_home"] == str(hermes_home)
    assert seen["memory_dir"] == str(hermes_home / "memories")
    assert seen["platform"] == "telegram"
    assert seen["session_id"] == agent.session_id
    assert seen["projection_source"] == "runtime/memory/MEMORY.md"
    assert seen["required"] is False
    # Never ship secret values through the seam.
    assert "api_key" not in seen


def test_gate_reprojection_lands_in_the_frozen_snapshot(
    monkeypatch, hermes_home, register_gate
):
    """The seam is before the freeze: bytes written by the gate are the ones loaded."""

    def _project(**kwargs):
        Path(kwargs["memory_dir"], "MEMORY.md").write_text(
            "task06-freshly-projected\n", encoding="utf-8"
        )
        return {"action": "allow"}

    register_gate(_project)
    agent = _make_agent(monkeypatch)

    block = agent._memory_store.format_for_system_prompt("memory")
    assert "task06-freshly-projected" in block
    assert "task06-memory-line" not in block


@pytest.mark.parametrize(
    "platform,skip_memory,enabled_toolsets",
    [
        ("cli", False, None),          # CLI / query construction
        ("telegram", False, None),     # gateway construction
        ("cron", False, ["memory"]),   # cron construction (E10 shape)
        ("cli", True, ["memory"]),     # flush/background w/ memory toolset (#65429)
    ],
)
def test_gate_fires_on_every_construction_path(
    monkeypatch, hermes_home, register_gate, platform, skip_memory, enabled_toolsets
):
    calls = []
    register_gate(lambda **kw: calls.append(kw.get("platform")) or {"action": "allow"})
    _make_agent(
        monkeypatch,
        platform=platform,
        skip_memory=skip_memory,
        enabled_toolsets=enabled_toolsets,
    )
    assert calls == [platform]


def test_gate_does_not_fire_when_no_memory_load_is_attempted(
    monkeypatch, hermes_home, register_gate
):
    """skip_memory + memory denylisted: nothing loads, so nothing to gate."""
    calls = []
    register_gate(lambda **kw: calls.append(kw) or {"action": "allow"})
    agent = _make_agent(
        monkeypatch,
        skip_memory=True,
        enabled_toolsets=["memory", "file"],
        disabled_toolsets=["memory"],
    )
    assert agent._memory_store is None
    assert calls == []


# ---------------------------------------------------------------------------
# 4. Fail-closed paths — gate outcomes always abort init loudly
# ---------------------------------------------------------------------------

def test_raising_hook_aborts_init_loudly(monkeypatch, hermes_home, register_gate):
    def boom(**_kwargs):
        raise RuntimeError("projection plugin is down")

    register_gate(boom)
    with pytest.raises(plugins_mod.PreMemoryLoadBlocked) as exc:
        _make_agent(monkeypatch)
    assert "pre_memory_load" in str(exc.value)
    assert "projection plugin is down" in str(exc.value)


def test_blocking_hook_aborts_init(monkeypatch, hermes_home, register_gate):
    register_gate(lambda **kw: {"action": "block", "message": "stale projection"})
    with pytest.raises(plugins_mod.PreMemoryLoadBlocked) as exc:
        _make_agent(monkeypatch)
    assert "stale projection" in str(exc.value)


def test_claude_code_block_shape_aborts_init(monkeypatch, hermes_home, register_gate):
    register_gate(lambda **kw: {"decision": "block", "reason": "watermark behind"})
    with pytest.raises(plugins_mod.PreMemoryLoadBlocked) as exc:
        _make_agent(monkeypatch)
    assert "watermark behind" in str(exc.value)


def test_timeout_hook_aborts_init(monkeypatch, hermes_home, register_gate):
    monkeypatch.setattr(plugins_mod, "_resolve_hook_callback_timeout", lambda: 0.1)
    hold = threading.Event()

    def hung(**_kwargs):
        hold.wait(timeout=10.0)
        return {"action": "allow"}

    register_gate(hung)
    try:
        with pytest.raises(plugins_mod.PreMemoryLoadBlocked) as exc:
            _make_agent(monkeypatch)
        assert "timed out" in str(exc.value)
    finally:
        hold.set()


def test_gate_failure_is_not_swallowed_by_the_memory_except(
    monkeypatch, hermes_home, register_gate
):
    """Delta 1 behavioral proof: a broken store must not mask a gate failure."""
    from tools.memory_tool import MemoryStore

    def _boom_load(self, *a, **kw):
        raise OSError("MEMORY.md unreadable")

    monkeypatch.setattr(MemoryStore, "load_from_disk", _boom_load)

    def boom(**_kwargs):
        raise RuntimeError("gate refused")

    register_gate(boom)
    with pytest.raises(plugins_mod.PreMemoryLoadBlocked):
        _make_agent(monkeypatch)


def test_blocked_gate_never_loads_memory(monkeypatch, hermes_home, register_gate):
    from tools.memory_tool import MemoryStore

    loads = []
    real_load = MemoryStore.load_from_disk
    monkeypatch.setattr(
        MemoryStore,
        "load_from_disk",
        lambda self, *a, **kw: loads.append(1) or real_load(self, *a, **kw),
    )
    register_gate(lambda **kw: {"action": "block", "message": "no"})
    with pytest.raises(plugins_mod.PreMemoryLoadBlocked):
        _make_agent(monkeypatch)
    assert loads == []


# ---------------------------------------------------------------------------
# 5. Delta 3 — the two failure classes, named per profile
# ---------------------------------------------------------------------------

def test_store_load_fault_is_still_swallowed_by_default(monkeypatch, hermes_home):
    """Upstream default: a genuine MemoryStore load fault runs empty, not loud."""
    from tools.memory_tool import MemoryStore

    monkeypatch.setattr(
        MemoryStore,
        "load_from_disk",
        lambda self, *a, **kw: (_ for _ in ()).throw(OSError("MEMORY.md unreadable")),
    )
    agent = _make_agent(monkeypatch)  # must not raise
    assert agent._memory_store is not None


def test_store_load_fault_fails_loud_when_the_gate_is_required(
    monkeypatch, hermes_home, register_gate
):
    _write_memory_config(hermes_home, pre_memory_load_required=True)
    from tools.memory_tool import MemoryStore

    monkeypatch.setattr(
        MemoryStore,
        "load_from_disk",
        lambda self, *a, **kw: (_ for _ in ()).throw(OSError("MEMORY.md unreadable")),
    )
    register_gate(lambda **kw: {"action": "allow"})

    from agent.agent_init import MemoryLoadFailed

    with pytest.raises(MemoryLoadFailed) as exc:
        _make_agent(monkeypatch)
    assert "MEMORY.md unreadable" in str(exc.value)


def test_required_gate_with_no_registered_hook_aborts(monkeypatch, hermes_home):
    """`required` means a projection owner must answer — silence is not allow."""
    _write_memory_config(hermes_home, pre_memory_load_required=True)
    with pytest.raises(plugins_mod.PreMemoryLoadBlocked) as exc:
        _make_agent(monkeypatch)
    assert "required" in str(exc.value)


def test_required_gate_needs_an_explicit_allow(monkeypatch, hermes_home, register_gate):
    _write_memory_config(hermes_home, pre_memory_load_required=True)
    register_gate(lambda **kw: None)
    with pytest.raises(plugins_mod.PreMemoryLoadBlocked):
        _make_agent(monkeypatch)


def test_required_gate_allows_when_hook_allows(monkeypatch, hermes_home, register_gate):
    _write_memory_config(hermes_home, pre_memory_load_required=True)
    register_gate(lambda **kw: {"action": "allow"})
    agent = _make_agent(monkeypatch)
    assert agent._memory_store is not None
    assert "task06-memory-line" in agent._memory_store.format_for_system_prompt("memory")


# ---------------------------------------------------------------------------
# 6. Red-team F1 - BaseException smuggling must fail closed
# ---------------------------------------------------------------------------
# invoke_hook's timeout worker used to catch only ``except Exception``. A gate
# callback raising SystemExit / KeyboardInterrupt / any BaseException killed
# the worker thread, ``finally`` still set ``done``, and the callback
# contributed nothing - the gate ALLOWED and memory loaded. A plugin that
# calls ``sys.exit()`` or validates its config with ``argparse`` is a
# realistic trigger. The worker must catch BaseException and hand it to the
# caller thread, where the hook's error policy applies.

class _CustomBaseException(BaseException):
    pass


@pytest.mark.parametrize(
    "smuggled",
    [
        SystemExit("projection stale"),
        KeyboardInterrupt(),
        _CustomBaseException("custom base"),
    ],
    ids=["system_exit", "keyboard_interrupt", "custom_base_exception"],
)
def test_base_exception_callback_yields_block_directive_from_invoke_hook(smuggled):
    mgr = PluginManager()
    mgr._discovered = True

    def boom(**_kwargs):
        raise smuggled

    mgr._hooks[HOOK] = [boom]
    results = mgr.invoke_hook(HOOK)
    assert results and results[0].get("action") == "block"
    assert HOOK in results[0].get("message", "")


def test_system_exit_gate_aborts_init_loudly(monkeypatch, hermes_home, register_gate):
    """End-to-end: a sys.exit()-style gate callback must veto the load."""

    def boom(**_kwargs):
        raise SystemExit("projection stale")

    register_gate(boom)
    with pytest.raises(plugins_mod.PreMemoryLoadBlocked) as exc:
        _make_agent(monkeypatch)
    assert HOOK in str(exc.value)


def test_pre_tool_call_base_exception_is_logged_not_swallowed(caplog):
    """pre_tool_call keeps its historical fail-open-on-raise policy, but a
    BaseException must be logged like any raised callback - not die silently
    on the worker thread."""
    import logging

    mgr = PluginManager()
    mgr._discovered = True

    def boom(**_kwargs):
        raise SystemExit("tool gate exit")

    mgr._hooks["pre_tool_call"] = [boom]
    with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
        results = mgr.invoke_hook("pre_tool_call", tool_name="x")
    assert results == []  # historical fail-open outcome, unchanged
    assert any("tool gate exit" in record.getMessage() for record in caplog.records)


# ---------------------------------------------------------------------------
# 7. Red-team F3 - `required` flag uses the shared truthy coercion
# ---------------------------------------------------------------------------
# agent_init parsed memory.pre_memory_load_required with raw bool(), so a
# quoted YAML string like "false" / "no" / "0" ENABLED required mode. Use the
# already-imported is_truthy_value instead.

def test_quoted_false_string_does_not_enable_required_mode(
    monkeypatch, hermes_home
):
    _write_memory_config(hermes_home, pre_memory_load_required="false")
    agent = _make_agent(monkeypatch)  # must not raise PreMemoryLoadBlocked
    assert agent._memory_store is not None


@pytest.mark.parametrize("off", ["no", "0", "off", ""])
def test_other_falsey_strings_do_not_enable_required_mode(
    monkeypatch, hermes_home, off
):
    _write_memory_config(hermes_home, pre_memory_load_required=off)
    agent = _make_agent(monkeypatch)
    assert agent._memory_store is not None


@pytest.mark.parametrize("on", ["true", "yes", "1"])
def test_truthy_strings_enable_required_mode(monkeypatch, hermes_home, on):
    _write_memory_config(hermes_home, pre_memory_load_required=on)
    with pytest.raises(plugins_mod.PreMemoryLoadBlocked) as exc:
        _make_agent(monkeypatch)
    assert "required" in str(exc.value)


# ---------------------------------------------------------------------------
# 8. Issue #41 — every later MEMORY.md freeze is gated too
# ---------------------------------------------------------------------------

def test_compaction_reload_gate_blocks_before_memory_is_refrozen(
    monkeypatch, hermes_home, register_gate
):
    """A gate that turns unhealthy after startup must veto compaction reload."""
    gate_calls = []

    def _gate(**kwargs):
        gate_calls.append(kwargs)
        if len(gate_calls) == 1:
            return {"action": "allow"}
        return {"action": "block", "message": "projection became stale"}

    register_gate(_gate)
    agent = _make_agent(monkeypatch)
    memory_store = getattr(agent, "_memory_store")

    loads = []
    monkeypatch.setattr(
        memory_store,
        "load_from_disk",
        lambda: loads.append("unguarded reload"),
    )

    with pytest.raises(plugins_mod.PreMemoryLoadBlocked) as exc:
        agent._invalidate_system_prompt()

    assert "projection became stale" in str(exc.value)
    assert len(gate_calls) == 2
    assert loads == []


def test_compaction_reload_refreezes_bytes_written_by_the_gate(
    monkeypatch, hermes_home, register_gate
):
    """The reload seam stays before load_from_disk, just like construction."""
    gate_calls = 0

    def _project(**kwargs):
        nonlocal gate_calls
        gate_calls += 1
        if gate_calls == 2:
            Path(kwargs["memory_dir"], "MEMORY.md").write_text(
                "task06-reprojected-at-compaction\n", encoding="utf-8"
            )
        return {"action": "allow"}

    register_gate(_project)
    agent = _make_agent(monkeypatch)
    memory_store = getattr(agent, "_memory_store")
    assert "task06-memory-line" in memory_store.format_for_system_prompt("memory")

    agent._invalidate_system_prompt()

    block = memory_store.format_for_system_prompt("memory")
    assert gate_calls == 2
    assert "task06-reprojected-at-compaction" in block
    assert "task06-memory-line" not in block
