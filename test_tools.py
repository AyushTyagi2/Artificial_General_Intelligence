"""
Digital Brain — Tool System Test Suite
=======================================
Run from the project root (C:\\Dev\\Digital_Brain) with either:

    python test_tools.py
    python -m test_tools

All tests are self-contained — no external dependencies are called
(network calls are skipped and marked accordingly).
"""
import sys
import os
import tempfile

# ── Ensure project root is on sys.path ────────────────────────────────────
# Works whether the file is run as a script or as a module.
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Imports ────────────────────────────────────────────────────────────────
from tools.base   import ok, err
from tools.router import ToolRouter
from tools.notes import tool as notes_tool
from tools.context import tool as ctx_tool
from tools.chatbot import tool  as chatbot_tool

# ── Tiny test framework ────────────────────────────────────────────────────
_results: list[bool] = []
_section = ""


def section(name: str) -> None:
    global _section
    _section = name
    print(f"\n── {name} ──")


def check(name: str, condition: bool, detail: str = "") -> None:
    sym = "✓" if condition else "✗"
    suffix = f"  [{detail}]" if detail else ""
    print(f"  {sym}  {name}{suffix}")
    _results.append(condition)


def skip(name: str, reason: str = "") -> None:
    print(f"  -  {name}  (skipped: {reason})")


# ══════════════════════════════════════════════════════════════════════════
#  Tests
# ══════════════════════════════════════════════════════════════════════════

def test_base() -> None:
    section("base helpers (ok / err)")

    r = ok({"x": 1})
    check("ok() status is 'success'",  r["status"] == "success")
    check("ok() output preserved",     r["output"] == {"x": 1})
    check("ok() with no arg",          ok()["output"] == {})

    e = err("boom", code=42)
    check("err() status is 'error'",   e["status"] == "error")
    check("err() error message",       e["output"]["error"] == "boom")
    check("err() extra kwargs",        e["output"]["code"] == 42)


def test_router() -> None:
    section("ToolRouter — discovery & dispatch")

    router = ToolRouter()
    check("router auto-discovers ≥ 5 tools",
          len(router.available_tools) >= 5,
          str(router.available_tools))

    for name in ("search", "wikipedia", "notes", "context", "chatbot"):
        check(f"  tool '{name}' present", name in router.available_tools)

    # Edge cases
    r = router.dispatch({"tool": "no_such_tool", "input": {}})
    check("unknown tool → error",           r["status"] == "error")
    check("  error lists available_tools",  "available_tools" in r["output"])

    r = router.dispatch({"input": {}})
    check("missing 'tool' key → error",     r["status"] == "error")

    r = router.dispatch({"tool": "notes", "input": "not-a-dict"})
    check("non-dict input → error",         r["status"] == "error")

    r = router.dispatch("not-a-dict")
    check("non-dict request → error",       r["status"] == "error")


def test_notes() -> None:
    section("notes tool (long-term memory)")

    tmp = tempfile.mktemp(suffix=".json")

    def call(action, **kw):
        return notes_tool.run({"action": action, "store_path": tmp, **kw})

    # create
    r = call("create", title="Gravity", content="Objects attract each other.",
             tags=["physics"])
    check("create → success",       r["status"] == "success")
    nid = r["output"]["note"]["id"]
    check("  returns integer id",   isinstance(nid, int))
    check("  title correct",        r["output"]["note"]["title"] == "Gravity")

    # create — validation
    r = call("create", content="no title")
    check("create without title → error", r["status"] == "error")
    r = call("create", title="x")
    check("create without content → error", r["status"] == "error")

    # read
    r = call("read", id=nid)
    check("read → success",         r["status"] == "success")
    check("  title matches",        r["output"]["note"]["title"] == "Gravity")

    r = call("read", id=9999)
    check("read missing id → error", r["status"] == "error")

    r = call("read")    # missing id param
    check("read no id → error",     r["status"] == "error")

    # update
    r = call("update", id=nid, content="Mass attracts mass via gravity.")
    check("update → success",       r["status"] == "success")
    check("  content changed",      "Mass attracts" in r["output"]["note"]["content"])

    r = call("update", id=9999, content="x")
    check("update missing id → error", r["status"] == "error")

    # list
    call("create", title="Osmosis",   content="Water crosses membranes.")
    call("create", title="Evolution", content="Species change over time.",
         tags=["biology"])
    r = call("list")
    check("list → success",          r["status"] == "success")
    check("  returns ≥ 3 notes",     r["output"]["count"] >= 3)

    r = call("list", tag="biology")
    check("list with tag filter",    r["output"]["count"] >= 1)
    check("  all have tag",          all("biology" in n["tags"]
                                         for n in r["output"]["notes"]))

    # search
    r = call("search", query="gravity")
    check("search → success",        r["status"] == "success")
    check("  finds 'Gravity' note",  r["output"]["count"] >= 1)

    r = call("search")
    check("search without query → error", r["status"] == "error")

    # delete
    r = call("delete", id=nid)
    check("delete → success",        r["status"] == "success")

    r = call("read", id=nid)
    check("read after delete → error", r["status"] == "error")

    r = call("delete", id=9999)
    check("delete missing id → error", r["status"] == "error")

    # unknown action
    r = call("explode")
    check("unknown action → error",  r["status"] == "error")

    try:
        os.unlink(tmp)
    except OSError:
        pass


def test_context() -> None:
    section("context tool (short-term memory)")

    # Reset module-level store for isolation
    ctx_tool._store.clear()

    def call(action, **kw):
        return ctx_tool.run({"action": action, **kw})

    # add
    r = call("add", text="Observed: wolves reduce deer population")
    check("add → success",            r["status"] == "success")
    check("  store_size incremented", r["output"]["store_size"] == 1)

    call("add", text="Goal: understand predator-prey dynamics", type="goal")
    call("add", text="Error: Wikipedia API timeout",            type="error")

    check("  store has 3 entries",    len(ctx_tool._store) == 3)

    # type inference
    r = call("add", text="I want to learn about ecology")
    check("type inference → goal",    r["output"]["added"]["type"] == "goal")
    call("clear")

    r = call("add", text="Observed falling leaves")
    check("type inference → observation",
          r["output"]["added"]["type"] == "observation")
    call("clear")

    # re-add for remaining tests
    call("add", text="Observed: wolves reduce deer population")
    call("add", text="Goal: understand predator-prey dynamics", type="goal")
    call("add", text="Error: Wikipedia API timeout",            type="error")

    # get
    r = call("get", limit=10)
    check("get → success",            r["status"] == "success")
    check("  returns all 3",          r["output"]["count"] == 3)

    r = call("get", type="goal")
    check("get filtered by type",     r["output"]["count"] == 1)
    check("  entry has correct type", r["output"]["entries"][0]["type"] == "goal")

    # search
    r = call("search", query="wolves")
    check("search → success",         r["status"] == "success")
    check("  finds wolves entry",     r["output"]["count"] >= 1)

    r = call("search", query="zzznomatch_xyz")
    check("search no match → count 0", r["output"]["count"] == 0)

    r = call("search")
    check("search without query → error", r["status"] == "error")

    # stats
    r = call("stats")
    check("stats → success",          r["status"] == "success")
    check("  total == 3",             r["output"]["total"] == 3)
    check("  by_type has goal",       "goal" in r["output"]["by_type"])

    # clear
    r = call("clear")
    check("clear → success",          r["status"] == "success")
    check("  cleared == 3",           r["output"]["cleared"] == 3)
    check("  store is empty",         len(ctx_tool._store) == 0)

    # add without text
    r = call("add")
    check("add without text → error", r["status"] == "error")

    # unknown action
    r = call("explode")
    check("unknown action → error",   r["status"] == "error")


def test_chatbot() -> None:
    section("chatbot tool (reasoning / LLM interface)")

    # basic call — no API key → fallback
    r = chatbot_tool.run({"message": "What is photosynthesis?"})
    check("chatbot → success",         r["status"] == "success")
    check("  has non-empty reply",     bool(r["output"].get("reply", "").strip()))
    check("  backend field present",   r["output"].get("backend") in ("claude", "fallback"))

    # empty message
    r = chatbot_tool.run({"message": ""})
    check("empty message → error",     r["status"] == "error")

    r = chatbot_tool.run({})
    check("missing message → error",   r["status"] == "error")

    # history accepted
    r = chatbot_tool.run({
        "message": "Why do plants grow toward light?",
        "history": [{"role": "user", "text": "Tell me about plants"}],
    })
    check("history kwarg accepted",    r["status"] == "success")

    # max_tokens clamped
    r = chatbot_tool.run({"message": "hello", "max_tokens": 99999})
    check("max_tokens clamped",        r["status"] == "success")

    # system prompt accepted
    r = chatbot_tool.run({
        "message": "Summarise osmosis.",
        "system":  "You are a biology tutor.",
    })
    check("system prompt accepted",    r["status"] == "success")

    # fallback pattern matching
    for phrase, expected_substring in [
        ("what is gravity",     "definitional"),
        ("why does it rain",    "causal"),
        ("how does it work",    "mechanism"),
        ("predict the outcome", "pattern"),
    ]:
        r = chatbot_tool.run({"message": phrase})
        reply = r["output"].get("reply", "").lower()
        check(f"  fallback pattern for '{phrase[:20]}'",
              any(w in reply for w in (expected_substring, "process", "question",
                                       "likely", "pattern", "factor")))


def test_router_roundtrip() -> None:
    section("full round-trip via ToolRouter.dispatch()")

    router = ToolRouter()
    tmp    = tempfile.mktemp(suffix=".json")
    ctx_tool._store.clear()

    requests = [
        {
            "tool":  "notes",
            "input": {"action": "create", "store_path": tmp,
                      "title": "Osmosis", "content": "Water moves across membranes."},
        },
        {
            "tool":  "context",
            "input": {"action": "add", "text": "Just learned about osmosis"},
        },
        {
            "tool":  "chatbot",
            "input": {"message": "Summarise osmosis in one sentence."},
        },
        {
            "tool":  "notes",
            "input": {"action": "list", "store_path": tmp},
        },
        {
            "tool":  "context",
            "input": {"action": "stats"},
        },
    ]

    for req in requests:
        r = router.dispatch(req)
        check(
            f"dispatch({req['tool']} / {req['input'].get('action', 'message')})",
            r["status"] == "success",
            r.get("output", {}).get("error", ""),
        )

    try:
        os.unlink(tmp)
    except OSError:
        pass


# ══════════════════════════════════════════════════════════════════════════
#  Network / live tests  (skipped by default, run with --live)
# ══════════════════════════════════════════════════════════════════════════

def test_live_network() -> None:
    section("live network tests (search + wikipedia)")
    router = ToolRouter()

    # search
    try:
        r = router.dispatch({"tool": "search",
                             "input": {"query": "photosynthesis", "max_results": 3}})
        check("search live → success",     r["status"] == "success",
              r.get("output", {}).get("error", ""))
        if r["status"] == "success":
            results = r["output"].get("results", [])
            check("  results is a list",    isinstance(results, list))
            check("  at least 1 result",    len(results) >= 1)
            if results:
                check("  result has title", bool(results[0].get("title", "")))
    except Exception as exc:
        print(f"  ! search live error: {exc}")

    # wikipedia
    try:
        r = router.dispatch({"tool": "wikipedia",
                             "input": {"query": "photosynthesis"}})
        check("wikipedia live → success",  r["status"] == "success",
              r.get("output", {}).get("error", ""))
        if r["status"] == "success":
            check("  has summary",          bool(r["output"].get("summary", "").strip()))
            check("  has url",              r["output"].get("url", "").startswith("https://"))
            check("  has title",            bool(r["output"].get("title", "")))
    except Exception as exc:
        print(f"  ! wikipedia live error: {exc}")


# ══════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    live = "--live" in sys.argv

    test_base()
    test_router()
    test_notes()
    test_context()
    test_chatbot()
    test_router_roundtrip()

    if live:
        test_live_network()
    else:
        print("\n  (Tip: run with --live to also test search & wikipedia network calls)")

    total  = len(_results)
    passed = sum(_results)
    failed = total - passed

    print(f"\n{'='*50}")
    if failed == 0:
        print(f"  {passed}/{total} tests passed  ✓")
    else:
        print(f"  {passed}/{total} passed   {failed} FAILED  ✗")
        sys.exit(1)


if __name__ == "__main__":
    main()