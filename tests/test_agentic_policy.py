from vllm_mlx.agentic_policy import classify_agentic_phase


def test_agentic_policy_classifies_initial_scaffold():
    messages = [
        {"role": "user", "content": "create a REST API"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
    ]

    assert classify_agentic_phase(messages, tools_requested=True) == "initial_scaffold"


def test_agentic_policy_classifies_repair_after_failure():
    messages = [
        {"role": "user", "content": "create a REST API"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "bun test\n1 fail\nerror: beforeEach is not defined"},
    ]

    assert classify_agentic_phase(messages, tools_requested=True) == "repair"


def test_agentic_policy_classifies_finalization_after_tests_pass():
    messages = [
        {"role": "user", "content": "create a REST API"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "bun test\n22 pass\n0 fail"},
    ]

    assert classify_agentic_phase(messages, tools_requested=True) == "finalization"

