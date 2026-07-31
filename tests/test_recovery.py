"""Getting a work order back, and knowing how many are still coming.

Calls are processed one at a time on purpose, so a burst can leave a work
order appearing ten minutes after the call it came from. And the popup is not
the record -- closing it must not lose anything.
"""

import time

import pytest

from call_transcriber import config, pipeline, storage


@pytest.fixture
def cfg(tmp_path):
    return config.load(tmp_path / "no-such-config.toml", root=tmp_path)


def write_call(root, when, name, order, transcript="hello", extracted=None):
    folder = storage.call_dir(root, when, name)
    storage.save_results(
        folder, order, transcript,
        extracted or {"caller_name": name.title(), "issue_summary": "Door stuck"},
    )
    # call_dir names by timestamp, but latest_work_order sorts by mtime.
    for path in folder.iterdir():
        import os
        os.utime(path, (when, when))
    return folder


# -- finding the last one --------------------------------------------------


def test_nothing_recorded_yet_is_not_an_error(tmp_path):
    assert storage.latest_work_order(tmp_path) is None
    assert storage.latest_work_order(tmp_path / "never-existed") is None


def test_the_newest_work_order_wins(cfg):
    base = time.mktime((2026, 7, 31, 9, 0, 0, 0, 0, -1))
    write_call(cfg.output_dir, base, "early", "FIRST ORDER")
    write_call(cfg.output_dir, base + 7200, "late", "SECOND ORDER")

    found = storage.latest_work_order(cfg.output_dir)
    assert found.read_text(encoding="utf-8") == "SECOND ORDER"


def test_a_call_reads_back_whole(cfg):
    when = time.mktime((2026, 7, 31, 9, 0, 0, 0, 0, -1))
    folder = write_call(
        cfg.output_dir, when, "rumi", "WORK ORDER\nCUSTOMER  Rumi",
        transcript="Rumi here, my door will not open.",
        extracted={"caller_name": "Rumi", "issue_summary": "Door will not open"},
    )

    call = storage.read_call(folder)
    assert "Rumi" in call["work_order"]
    assert "will not open" in call["transcript"]
    assert call["extracted"]["caller_name"] == "Rumi"


def test_a_corrupt_extracted_file_does_not_stop_the_rest_loading(cfg):
    when = time.mktime((2026, 7, 31, 9, 0, 0, 0, 0, -1))
    folder = write_call(cfg.output_dir, when, "rumi", "WORK ORDER")
    (folder / "extracted.json").write_text("{ not json", encoding="utf-8")

    call = storage.read_call(folder)
    assert call["work_order"] == "WORK ORDER"
    assert call["extracted"] == {}


# -- putting it back on screen ---------------------------------------------


class Ui:
    def __init__(self):
        self.posted = []

    def request(self, popup):
        self.posted.append(popup)


def test_closing_the_window_does_not_lose_the_call(cfg, monkeypatch):
    monkeypatch.setattr(pipeline.notify, "copy", lambda text: True)
    when = time.mktime((2026, 7, 31, 9, 0, 0, 0, 0, -1))
    write_call(cfg.output_dir, when, "rumi", "WORK ORDER\nCUSTOMER  Rumi")

    ui = Ui()
    runner = pipeline.Runner(cfg, ui=ui)

    assert runner.show_last_work_order() is True
    assert len(ui.posted) == 1
    assert "Rumi" in ui.posted[0].work_order
    assert ui.posted[0].copied is True
    assert ui.posted[0].folder.is_dir()


def test_reopening_says_so_when_there_is_nothing_to_reopen(cfg):
    ui = Ui()
    runner = pipeline.Runner(cfg, ui=ui)

    assert runner.show_last_work_order() is False
    assert ui.posted == []


# -- the queue -------------------------------------------------------------


def test_a_burst_of_calls_queues_rather_than_dropping_any(cfg, monkeypatch):
    """Five calls back to back: none lost, and the depth is visible."""
    processed = []
    monkeypatch.setattr(
        pipeline, "process_call", lambda call, cfg, ui=None: processed.append(call)
    )

    runner = pipeline.Runner(cfg)
    calls = [
        pipeline.Call(
            audio=__import__("numpy").zeros((16000, 1), dtype="int16"),
            sample_rate=16000, started_at=1785500000.0 + i,
            duration_s=1.0, ended_reason="manual",
        )
        for i in range(5)
    ]
    for call in calls:
        runner._submit(call)

    runner._pool.shutdown(wait=True)

    assert len(processed) == 5
    assert runner.calls_handled == 5
    assert runner.pending == 0


def test_the_queue_depth_is_reported_while_work_is_waiting(cfg, monkeypatch):
    import threading

    release = threading.Event()
    monkeypatch.setattr(
        pipeline, "process_call", lambda call, cfg, ui=None: release.wait(5)
    )

    runner = pipeline.Runner(cfg)
    call = pipeline.Call(
        audio=__import__("numpy").zeros((16000, 1), dtype="int16"),
        sample_rate=16000, started_at=1785500000.0, duration_s=1.0,
        ended_reason="manual",
    )
    for _ in range(3):
        runner._submit(call)

    assert runner.pending == 3

    release.set()
    runner._pool.shutdown(wait=True)
    assert runner.pending == 0


def test_a_call_that_fails_still_leaves_the_queue(cfg, monkeypatch):
    """Otherwise the tray would report work pending forever."""
    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline, "process_call", explode)

    runner = pipeline.Runner(cfg)
    runner._submit(pipeline.Call(
        audio=__import__("numpy").zeros((16000, 1), dtype="int16"),
        sample_rate=16000, started_at=1785500000.0, duration_s=1.0,
        ended_reason="manual",
    ))
    runner._pool.shutdown(wait=True)

    assert runner.pending == 0
