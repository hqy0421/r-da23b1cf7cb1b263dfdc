import multiprocessing
import threading
import time
import typing as tp
from dataclasses import dataclass

from pypeln import utils as pypeln_utils

from .queue import IterableQueue
from .worker import Worker


@dataclass
class Supervisor:
    workers: tp.List[Worker]
    main_queue: IterableQueue
    done: bool = False

    def __call__(self):
        try:
            # the worker list is re-read on every pass so that workers added
            # at runtime (through a WorkerController) are also supervised
            while not self.done:
                for worker in list(self.workers):
                    if worker.timeout > 0 and worker.did_timeout():
                        worker.stop()
                        worker.start()

                time.sleep(0.01)

        except BaseException as e:
            self.main_queue.raise_exception(e)

    def __enter__(self):
        self.start()

    def __exit__(self, *args):
        self.done = True

        # end the work associated to this run: reject further adjustments
        for controller in self._controllers():
            controller.close()

        for worker in list(self.workers):
            worker.stop()

        while any(
            worker.process is not None and worker.process.is_alive()
            for worker in list(self.workers)
        ):
            time.sleep(pypeln_utils.TIMEOUT)

    def _controllers(self) -> tp.List[tp.Any]:
        controllers = []
        seen = set()

        for worker in list(self.workers):
            controller = getattr(worker.stage_params, "controller", None)

            if controller is not None and id(controller) not in seen:
                seen.add(id(controller))
                controllers.append(controller)

        return controllers

    def start(self):
        for worker in list(self.workers):
            worker.start()

        # activate the controllers once the pipeline is actually running so
        # runtime adjustments can be applied
        for controller in self._controllers():
            controller._activate(self)

        t = threading.Thread(target=self)
        t.daemon = True
        t.start()
