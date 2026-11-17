import threading
import time
import typing as tp
from unittest import TestCase

import pytest

import pypeln as pl

MAX_EXAMPLES = 10
T = tp.TypeVar("T")


def consume_in_background(stage, out, errors):
    def target():
        try:
            for x in stage:
                out.append(x)
        except BaseException as e:
            errors.append(e)

    t = threading.Thread(target=target, daemon=True)
    t.start()

    return t


class MyError(Exception):
    pass


class TestWorkerController(TestCase):
    def test_scale_up_adds_workers_with_their_own_resources(self):
        # new workers must run on_start on their own thread and receive
        # their own injected resources in f / on_done
        lock = threading.Lock()
        started = []
        finished = []
        seen = {}  # worker index -> list of (x, token)

        def on_start(worker_info):
            with lock:
                started.append(worker_info.index)

            return dict(token=f"token-{worker_info.index}")

        def on_done(worker_info, stage_status):
            with lock:
                finished.append(worker_info.index)

        def f(x, token, worker_info):
            with lock:
                seen.setdefault(worker_info.index, []).append((x, token))

            time.sleep(0.02)

            return x

        n = 30
        stage = pl.thread.map(
            f, range(n), workers=1, on_start=on_start, on_done=on_done
        )

        out, errors = [], []
        t = consume_in_background(stage, out, errors)

        assert stage.controller.wait_running(timeout=5)
        time.sleep(0.05)

        assert stage.controller.set_workers(3) == 3
        assert stage.controller.workers == 3

        t.join(30)
        assert not t.is_alive()
        assert not errors
        assert sorted(out) == list(range(n))

        # every worker (initial + added) built its own resources and cleaned up
        assert sorted(started) == [0, 1, 2]
        assert sorted(finished) == [0, 1, 2]

        # the added workers actually processed elements, and each element was
        # processed with the resource (token) of its own worker
        assert sorted(seen.keys()) == [0, 1, 2]
        assert len(seen[1]) > 0
        assert len(seen[2]) > 0

        for index, elems in seen.items():
            for x, token in elems:
                assert token == f"token-{index}"

        # exactly-once: processing counts add up to the number of elements
        snapshot = stage.controller.snapshot()
        assert sum(w["processed"] for w in snapshot) == n
        assert all(w["done"] for w in snapshot)

    def test_scale_down_retires_workers_without_new_records(self):
        # the retired worker must deliver its in-flight element, take no new
        # records afterwards, and exit normally (on_done runs on its thread)
        lock = threading.Lock()
        gate = threading.Event()
        both_in_flight = threading.Event()
        counts = {}
        finished = []

        def f(x, worker_info):
            with lock:
                counts.setdefault(worker_info.index, []).append(x)

                if sum(len(v) for v in counts.values()) == 2:
                    both_in_flight.set()

            gate.wait(10)

            return x

        def on_done(worker_info):
            with lock:
                finished.append(worker_info.index)

        n = 12
        stage = pl.thread.map(f, range(n), workers=2, on_done=on_done)

        out, errors = [], []
        t = consume_in_background(stage, out, errors)

        assert stage.controller.wait_running(timeout=5)
        assert both_in_flight.wait(10)

        # shrink to a positive number: the most recent worker is retired
        assert stage.controller.set_workers(1) == 1
        assert stage.controller.workers == 1

        gate.set()
        t.join(30)
        assert not t.is_alive()
        assert not errors

        # no element lost or duplicated
        assert sorted(out) == list(range(n))

        # the retired worker (index 1) only processed its in-flight element,
        # the surviving worker drained everything else
        assert len(counts[1]) == 1
        assert len(counts[0]) == n - 1

        # both workers exited normally
        assert sorted(finished) == [0, 1]

    def test_flat_map_retiring_worker_completes_expansion(self):
        # a retiring flat_map worker must deliver all the sub-items of its
        # current element before exiting
        lock = threading.Lock()
        gate = threading.Event()
        both_started = threading.Event()
        expansions = []

        def f(x, worker_info):
            with lock:
                expansions.append((worker_info.index, x))

                if len(expansions) == 2:
                    both_started.set()

            for i in range(3):
                gate.wait(10)
                yield (x, i)

        stage = pl.thread.flat_map(f, [0, 1], workers=2)

        out, errors = [], []
        t = consume_in_background(stage, out, errors)

        assert stage.controller.wait_running(timeout=5)
        assert both_started.wait(10)

        stage.controller.set_workers(1)
        gate.set()

        t.join(30)
        assert not t.is_alive()
        assert not errors

        # all sub-items of both in-flight expansions were delivered
        assert sorted(out) == [(x, i) for x in (0, 1) for i in range(3)]

    def test_element_index_and_processing_counts_preserved(self):
        indices = []
        lock = threading.Lock()

        def f(x, element_index):
            with lock:
                indices.append(element_index)

            time.sleep(0.01)

            return x

        n = 30
        stage = pl.thread.map(f, range(n), workers=1)

        out, errors = [], []
        t = consume_in_background(
            pl.thread.to_iterable(stage, return_index=True), out, errors
        )

        assert stage.controller.wait_running(timeout=5)
        time.sleep(0.05)
        stage.controller.set_workers(4)

        t.join(30)
        assert not t.is_alive()
        assert not errors

        # Element indices are preserved and each element was processed once
        assert sorted(elem.index for elem in out) == [(i,) for i in range(n)]
        assert sorted(elem.value for elem in out) == list(range(n))
        assert sorted(indices) == [(i,) for i in range(n)]

    def test_filter_stage_is_adjustable(self):
        def f(x):
            time.sleep(0.01)
            return x % 2 == 0

        n = 20
        stage = pl.thread.filter(f, range(n), workers=1)

        out, errors = [], []
        t = consume_in_background(stage, out, errors)

        assert stage.controller.wait_running(timeout=5)
        time.sleep(0.03)
        stage.controller.set_workers(3)

        t.join(30)
        assert not t.is_alive()
        assert not errors
        assert sorted(out) == [x for x in range(n) if x % 2 == 0]

    def test_flat_map_stage_is_adjustable(self):
        def f(x):
            time.sleep(0.01)
            yield x
            yield -x

        n = 10
        stage = pl.thread.flat_map(f, range(n), workers=1)

        out, errors = [], []
        t = consume_in_background(stage, out, errors)

        assert stage.controller.wait_running(timeout=5)
        time.sleep(0.03)
        stage.controller.set_workers(3)

        t.join(30)
        assert not t.is_alive()
        assert not errors
        assert sorted(out) == sorted([y for x in range(n) for y in (x, -x)])

    def test_results_feed_native_downstream(self):
        # the adjusted stage keeps working with native downstream stages
        def f(x):
            time.sleep(0.01)
            return x + 1

        n = 20
        source_stage = pl.thread.map(f, range(n), workers=1)
        stage = pl.thread.map(lambda x: x * 2, source_stage, workers=2)
        stage = pl.thread.ordered(stage)

        out, errors = [], []
        t = consume_in_background(stage, out, errors)

        assert source_stage.controller.wait_running(timeout=5)
        time.sleep(0.03)
        source_stage.controller.set_workers(3)

        t.join(30)
        assert not t.is_alive()
        assert not errors

        # ordered downstream restores the source order
        assert out == [(x + 1) * 2 for x in range(n)]

    def test_backpressure_is_respected(self):
        # queuing still backpressures per the maxsize configuration
        started = []
        out, errors = [], []

        def f(x):
            started.append(x)
            time.sleep(0.005)
            return x

        n = 15
        maxsize = 1
        stage = pl.thread.map(f, range(n), workers=1, maxsize=maxsize)

        def slow_consumer():
            try:
                for x in stage:
                    # elements inside the stage <= maxsize + in-flight workers
                    in_stage = len(started) - (len(out) + 1)
                    assert in_stage <= maxsize + stage.controller.active_workers + 1

                    time.sleep(0.02)
                    out.append(x)
            except BaseException as e:
                errors.append(e)

        t = threading.Thread(target=slow_consumer, daemon=True)
        t.start()

        assert stage.controller.wait_running(timeout=5)
        time.sleep(0.05)
        stage.controller.set_workers(3)

        t.join(30)
        assert not t.is_alive()
        assert not errors
        assert sorted(out) == list(range(n))

    def test_adjustment_racing_source_end(self):
        # adjustments concurrent with the end of the source must not cause
        # premature EOF, lose tail elements, or wait for missing producers
        for _ in range(5):
            n = 8

            def f(x):
                time.sleep(0.005 * (x % 3 + 1))
                return x

            stage = pl.thread.map(f, range(n), workers=2)

            out, errors = [], []
            t = consume_in_background(stage, out, errors)

            assert stage.controller.wait_running(timeout=5)

            target = 1

            while t.is_alive():
                target = target % 4 + 1

                try:
                    stage.controller.set_workers(target)
                except RuntimeError:
                    break

                time.sleep(0.001)

            t.join(30)
            assert not t.is_alive()
            assert not errors

            # no premature EOF and no tail loss: everything arrived exactly once
            assert sorted(out) == list(range(n))

            # all workers (initial + added) finished cleanly and every
            # element was processed exactly once
            snapshot = stage.controller.snapshot()
            assert all(w["done"] for w in snapshot)
            assert sum(w["processed"] for w in snapshot) == n

    def test_set_workers_validation(self):
        stage = pl.thread.map(lambda x: x, range(10), workers=1)

        with pytest.raises(ValueError):
            stage.controller.set_workers(0)

        with pytest.raises(ValueError):
            stage.controller.set_workers(-2)

        with pytest.raises(TypeError):
            stage.controller.set_workers(1.5)

        with pytest.raises(TypeError):
            stage.controller.set_workers(True)

        # the pipeline is not running yet
        assert not stage.controller.is_running
        assert not stage.controller.wait_running(timeout=0.05)

        with pytest.raises(RuntimeError):
            stage.controller.set_workers(2)

        # old behavior is unchanged
        assert sorted(list(stage)) == list(range(10))

    def test_adjustment_rejected_after_finish(self):
        stage = pl.thread.map(lambda x: x, range(5), workers=2)

        assert sorted(list(stage)) == list(range(5))

        with pytest.raises(RuntimeError):
            stage.controller.set_workers(3)

        assert stage.controller.closed

    def test_close_rejects_adjustments_but_pipeline_completes(self):
        def f(x):
            time.sleep(0.01)
            return x

        n = 10
        stage = pl.thread.map(f, range(n), workers=2)

        out, errors = [], []
        t = consume_in_background(stage, out, errors)

        assert stage.controller.wait_running(timeout=5)

        stage.controller.close()
        assert stage.controller.closed
        assert not stage.controller.is_running

        with pytest.raises(RuntimeError):
            stage.controller.set_workers(4)

        # closing the handle does not disturb the running pipeline
        t.join(30)
        assert not t.is_alive()
        assert not errors
        assert sorted(out) == list(range(n))

    def test_processing_failure_ends_adjustments(self):
        def f(x):
            if x == 3:
                raise MyError()

            return x

        stage = pl.thread.map(f, range(10), workers=2)

        with pytest.raises(MyError):
            list(stage)

        # the failure ended the work associated to this run
        assert stage.controller.closed

        with pytest.raises(RuntimeError):
            stage.controller.set_workers(3)

    def test_callback_failure_ends_adjustments(self):
        def on_start():
            raise MyError()

        stage = pl.thread.map(lambda x: x, range(10), workers=2, on_start=on_start)

        with pytest.raises(MyError):
            list(stage)

        assert stage.controller.closed

        with pytest.raises(RuntimeError):
            stage.controller.set_workers(3)

    def test_on_done_stage_status_stays_consistent(self):
        statuses = []
        lock = threading.Lock()

        def f(x):
            time.sleep(0.01)
            return x

        def on_done(stage_status):
            with lock:
                statuses.append((stage_status.active_workers, stage_status.done))

        n = 10
        stage = pl.thread.map(f, range(n), workers=2, on_done=on_done)

        out, errors = [], []
        t = consume_in_background(stage, out, errors)

        assert stage.controller.wait_running(timeout=5)
        time.sleep(0.03)
        stage.controller.set_workers(4)

        t.join(30)
        assert not t.is_alive()
        assert not errors
        assert sorted(out) == list(range(n))

        # on_done ran once per worker (initial + added), the counts never
        # went negative, and exactly one worker observed the stage as done
        assert len(statuses) == 4
        assert all(active >= 0 for active, _ in statuses)
        assert statuses.count((0, True)) == 1
        assert sum(1 for _, done in statuses if done) == 1

    def test_non_adjustable_stages_reject_adjustments(self):
        def f(x):
            time.sleep(0.01)
            return x

        n = 10
        source = pl.thread.from_iterable(range(n))
        stage = pl.thread.map(f, source, workers=1)
        ordered = pl.thread.ordered(stage)

        out, errors = [], []
        t = consume_in_background(ordered, out, errors)

        assert stage.controller.wait_running(timeout=5)

        # from_iterable and ordered cannot be adjusted safely
        with pytest.raises(RuntimeError):
            source.controller.set_workers(2)

        with pytest.raises(RuntimeError):
            ordered.controller.set_workers(2)

        # but the map stage in between can
        stage.controller.set_workers(2)

        t.join(30)
        assert not t.is_alive()
        assert not errors
        assert out == list(range(n))
