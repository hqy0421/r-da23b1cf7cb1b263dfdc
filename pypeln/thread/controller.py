import threading
import typing as tp

if tp.TYPE_CHECKING:
    from .stage import Stage
    from .supervisor import Supervisor
    from .worker import StageParams, Worker


class WorkerController:
    """
    Runtime handle used to adjust the number of workers of a running `pypeln.thread`
    stage (e.g. `map`, `filter`, `flat_map`) without re-reading the source or
    restarting the pipeline.

    Every adjustable `pl.thread.Stage` exposes its handle through the
    `stage.controller` attribute. The handle becomes active once the stage starts
    being consumed:

    ```python
    import pypeln as pl
    import threading

    stage = pl.thread.map(slow_fn, data, workers=2)
    stage = pl.thread.filter(slow_pred, stage, workers=1)

    results = []
    t = threading.Thread(target=lambda: results.extend(stage), daemon=True)
    t.start()

    # wait until the pipeline is actually running
    stage.controller.wait_running(timeout=5)

    # adjust the number of workers of the filter stage while it runs
    stage.controller.set_workers(4)  # scale up
    stage.controller.set_workers(1)  # scale down (must stay positive)

    t.join()
    ```

    Semantics:

    * Scaling up spawns brand new worker threads. Each new worker runs `on_start`
      on its own thread to build its own resources, and `f` / `on_done` receive
      the usual dependency-injection arguments (`worker_info`, `element_index`,
      `stage_status`, and the kwargs returned by `on_start`).
    * Scaling down marks existing workers as *retiring*. A retiring worker stops
      pulling new elements, delivers the element it is currently processing
      (including all the items of an ongoing `flat_map` expansion), runs
      `on_done`, notifies downstream, and exits normally on its own thread.
    * `Element` indices and the per-worker processing counts are preserved: every
      element is processed exactly once.
    * Adjustments that race with the end of the source never cause a premature
      EOF, never lose tail elements, and never leave downstream waiting for
      producers that do not exist.
    * If `f` or a callback fails, or when the pipeline ends, the controller is
      closed; `close()` can also be called explicitly. Adjustments are rejected
      with `RuntimeError` once the controller is closed.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._stage_params: tp.Optional["StageParams"] = None
        self._supervisor: tp.Optional["Supervisor"] = None
        self._workers: tp.List["Worker"] = []
        self._make_worker: tp.Optional[tp.Callable[[int], "Worker"]] = None
        self._next_index: int = 0
        self._target_workers: int = 0
        self._adjustable: bool = True
        self._closed: bool = False

    def __repr__(self) -> str:
        return f"WorkerController(workers={self.workers}, active_workers={self.active_workers}, closed={self.closed})"

    # ----------------------------------------------------------------
    # lifecycle hooks (used by the pipeline machinery)
    # ----------------------------------------------------------------

    def _bind(
        self,
        stage_params: "StageParams",
        workers: tp.List["Worker"],
        make_worker: tp.Callable[[int], "Worker"],
        adjustable: bool,
    ):
        with self._lock:
            self._stage_params = stage_params
            self._workers = list(workers)
            self._make_worker = make_worker
            self._next_index = len(self._workers)
            self._target_workers = len(self._workers)
            self._adjustable = adjustable
            self._supervisor = None
            self._closed = False
            self._running.clear()

    def _activate(self, supervisor: "Supervisor"):
        with self._lock:
            if self._closed:
                return

            self._supervisor = supervisor
            self._running.set()

    def _on_worker_done(
        self, stage_params: "StageParams", worker: tp.Optional["Worker"]
    ):
        # called by StageParams.worker_done every time a worker finishes its
        # processing loop, exactly once per worker
        with self._lock:
            with stage_params.namespace:
                stage_params.namespace.active_workers -= 1

            if stage_params is not self._stage_params:
                # belongs to a previous run of the stage, only its
                # active_workers count (updated above) is relevant
                return

            if worker is not None:
                worker.namespace.finished = True

            if worker is None or not worker.namespace.retiring:
                self._target_workers -= 1

    # ----------------------------------------------------------------
    # public API
    # ----------------------------------------------------------------

    def set_workers(self, workers: int) -> int:
        """
        Adjusts the number of workers of the stage to exactly `workers`.

        * If `workers` is greater than the current (non-retiring) worker count,
          new worker threads are created and started immediately.
        * If it is smaller, the exceeding workers are marked as retiring and
          wind down as described in the class documentation.

        Arguments:
            workers: The new number of workers, must be a positive integer.

        Returns:
            The new number of workers.

        Raises:
            TypeError: If `workers` is not an integer.
            ValueError: If `workers` is not a positive integer.
            RuntimeError: If the controller is closed, the stage is not
                running, the stage does not support adjustment, or the stage
                already finished.
        """
        if isinstance(workers, bool) or not isinstance(workers, int):
            raise TypeError(f"workers must be an int, got {type(workers).__name__}")

        if workers < 1:
            raise ValueError(f"workers must be a positive integer, got {workers}")

        with self._lock:
            if self._closed:
                raise RuntimeError("cannot adjust workers: the controller is closed")

            if self._stage_params is None or self._supervisor is None:
                raise RuntimeError("cannot adjust workers: the stage is not running")

            if not self._adjustable:
                raise RuntimeError(
                    "cannot adjust workers: this stage does not support worker adjustment"
                )

            current = self._target_workers

            if workers > current:
                if self._stage_params.namespace.active_workers == 0:
                    raise RuntimeError(
                        "cannot adjust workers: the stage has already finished"
                    )

                self._scale_up(workers - current)
            elif workers < current:
                self._scale_down(current - workers)

            return workers

    def close(self):
        """
        Closes the controller, ending its associated work: further adjustments
        are rejected with `RuntimeError`. The pipeline itself keeps running
        until it completes naturally. `close` is idempotent and is also called
        automatically when the pipeline ends or fails.
        """
        with self._lock:
            self._closed = True

    @property
    def closed(self) -> bool:
        """
        `bool`: `True` if the controller was closed (explicitly, or because the
        pipeline ended or failed) and no longer accepts adjustments.
        """
        with self._lock:
            return self._closed

    @property
    def is_running(self) -> bool:
        """
        `bool`: `True` if the stage is currently running and accepts adjustments.
        """
        with self._lock:
            return self._running.is_set() and not self._closed

    def wait_running(self, timeout: tp.Optional[float] = None) -> bool:
        """
        Blocks until the stage starts running (i.e. it is being consumed and
        adjustments are accepted) or `timeout` seconds elapse.

        Returns:
            `True` if the stage is running, `False` on timeout.
        """
        return self._running.wait(timeout)

    @property
    def workers(self) -> int:
        """
        `int`: The current number of (non-retiring) workers of the stage.
        """
        with self._lock:
            return self._target_workers

    @property
    def active_workers(self) -> int:
        """
        `int`: The number of workers currently alive, including retiring
        workers that are still winding down.
        """
        with self._lock:
            if self._stage_params is None:
                return 0

            return self._stage_params.namespace.active_workers

    def snapshot(self) -> tp.List[tp.Dict[str, tp.Any]]:
        """
        Returns a list with the state of every worker of the current run,
        including the workers added at runtime. Each entry contains the
        worker's `index`, the number of elements it has `processed`, and
        whether it is `retiring` or `done`.
        """
        with self._lock:
            return [
                dict(
                    index=worker.index,
                    processed=worker.namespace.processed,
                    retiring=worker.namespace.retiring,
                    done=worker.namespace.done,
                )
                for worker in self._workers
            ]

    # ----------------------------------------------------------------
    # scaling
    # ----------------------------------------------------------------

    def _scale_up(self, k: int):
        # must be called with self._lock held
        stage_params = self._stage_params

        # Account for the new workers *before* any of them can finish: each
        # worker sends exactly one DONE downstream when it exits, so the
        # downstream queues must expect `k` additional DONE signals. This
        # increment is atomic w.r.t. `worker_done` (both hold this lock), and
        # since at least one worker is still active, downstream cannot observe
        # `remaining == 0` (premature EOF) before the increment lands.
        for queue in stage_params.output_queues:
            with queue.namespace:
                queue.namespace.remaining += k

        with stage_params.namespace:
            stage_params.namespace.active_workers += k

        self._target_workers += k

        created: tp.List["Worker"] = []

        try:
            for _ in range(k):
                worker = self._make_worker(self._next_index)
                self._next_index += 1
                created.append(worker)
                self._workers.append(worker)
                self._supervisor.workers.append(worker)
                worker.start()
        except BaseException:
            # roll back the workers that never started so downstream never
            # waits for producers that do not exist
            unstarted_ids = {id(w) for w in created if w.process is None}
            missing = len(unstarted_ids)

            for queue in stage_params.output_queues:
                with queue.namespace:
                    queue.namespace.remaining -= missing

            with stage_params.namespace:
                stage_params.namespace.active_workers -= missing

            self._target_workers -= missing
            self._workers = [w for w in self._workers if id(w) not in unstarted_ids]
            self._supervisor.workers[:] = [
                w for w in self._supervisor.workers if id(w) not in unstarted_ids
            ]

            raise

    def _scale_down(self, k: int):
        # must be called with self._lock held. Retiring workers keep their
        # downstream DONE credit: they still deliver their current element and
        # exit normally, so no queue accounting needs to change here.
        candidates = [
            worker
            for worker in self._workers
            if not worker.namespace.retiring and not worker.namespace.finished
        ]

        # retire the most recently created workers first
        candidates.sort(key=lambda worker: worker.index, reverse=True)

        for worker in candidates[:k]:
            worker.retire()

        self._target_workers -= k
