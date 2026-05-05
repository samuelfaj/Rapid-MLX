import vllm_mlx.request_metrics as request_metrics
from vllm_mlx.request_metrics import RequestRecorder


def test_request_recorder_records_completed_request(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(request_metrics.time, "time", lambda: now[0])

    recorder = RequestRecorder()
    req_id = recorder.start("/v1/chat/completions")

    now[0] += 0.25
    recorder.mark_first_token(req_id)
    recorder.update(req_id, delta_text="hello", generated_tokens=1, prompt_tokens=8)

    now[0] += 0.75
    recorder.finish(
        req_id,
        finish_reason="stop",
        prompt_tokens=8,
        generated_tokens=4,
        engine_gen_tps=12.5,
        engine_ttft=0.25,
    )

    entries = recorder.entries()
    assert recorder.active() is None
    assert len(entries) == 1
    assert entries[0]["surface"] == "/v1/chat/completions"
    assert entries[0]["prompt_tokens"] == 8
    assert entries[0]["generated_tokens"] == 4
    assert entries[0]["decode_tps"] == 12.5
    assert entries[0]["ttft"] == 0.25


def test_request_recorder_falls_back_when_engine_tps_missing(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(request_metrics.time, "time", lambda: now[0])

    recorder = RequestRecorder()
    req_id = recorder.start("/v1/chat/completions")

    now[0] += 3.0
    recorder.mark_first_token(req_id)
    recorder.update(req_id, delta_text="first", generated_tokens=1, prompt_tokens=2000)

    now[0] += 2.0
    recorder.update(req_id, delta_text="rest", generated_tokens=100, prompt_tokens=2000)
    recorder.finish(
        req_id,
        finish_reason="stop",
        prompt_tokens=2000,
        generated_tokens=100,
        engine_ttft=3.0,
    )

    entry = recorder.entries()[0]
    assert entry["generated_tokens"] == 100
    assert entry["decode_tps"] == 50.0
    assert entry["generation_tps"] == 50.0


def test_request_recorder_prefers_engine_tps_over_text_chunk_timing(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(request_metrics.time, "time", lambda: now[0])

    recorder = RequestRecorder()
    req_id = recorder.start("/v1/chat/completions")

    now[0] += 1.0
    recorder.mark_first_token(req_id)
    recorder.update(req_id, delta_text="first", generated_tokens=1, prompt_tokens=100)

    now[0] += 3.0
    recorder.update(req_id, delta_text="last", generated_tokens=100, prompt_tokens=100)
    recorder.finish(
        req_id,
        finish_reason="tool_calls",
        prompt_tokens=100,
        generated_tokens=100,
        engine_gen_tps=60.0,
        engine_ttft=1.0,
    )

    entry = recorder.entries()[0]
    assert entry["decode_tps"] == 60.0
    assert entry["generation_tps"] == 60.0


def test_request_recorder_uses_last_token_time_for_decode_window(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(request_metrics.time, "time", lambda: now[0])

    recorder = RequestRecorder()
    req_id = recorder.start("/v1/chat/completions")

    now[0] += 1.0
    recorder.mark_first_token(req_id)
    recorder.update(req_id, delta_text="first", generated_tokens=1, prompt_tokens=100)

    now[0] += 1.0
    recorder.update(req_id, delta_text="last", generated_tokens=100, prompt_tokens=100)

    now[0] += 3.0
    recorder.finish(
        req_id,
        finish_reason="tool_calls",
        prompt_tokens=100,
        generated_tokens=100,
        engine_ttft=1.0,
    )

    entry = recorder.entries()[0]
    assert entry["elapsed"] == 5.0
    assert entry["decode_tps"] == 100.0
    assert entry["generation_tps"] == 100.0


def test_request_recorder_uses_engine_tps_when_no_text_chunks(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(request_metrics.time, "time", lambda: now[0])

    recorder = RequestRecorder()
    req_id = recorder.start("/v1/chat/completions")

    now[0] += 1.0
    recorder.finish(
        req_id,
        finish_reason="tool_calls",
        prompt_tokens=100,
        generated_tokens=100,
        engine_gen_tps=80.0,
        engine_ttft=0.2,
    )

    entry = recorder.entries()[0]
    assert entry["decode_tps"] == 80.0
    assert entry["generation_tps"] == 80.0


def test_request_recorder_active_snapshot(monkeypatch):
    monkeypatch.setattr(request_metrics.time, "time", lambda: 1000.0)

    recorder = RequestRecorder()
    req_id = recorder.start("/v1/completions")
    recorder.update(req_id, delta_text="partial", generated_tokens=2, prompt_tokens=5)

    active = recorder.active()
    assert active is not None
    assert active["request_id"] == req_id
    assert active["surface"] == "/v1/completions"
    assert active["generated_tokens"] == 2
    assert "partial" in active["message_preview"]
