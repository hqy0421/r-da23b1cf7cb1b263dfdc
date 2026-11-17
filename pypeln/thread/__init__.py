"""
The `thread` module lets you create pipelines using objects from python's [threading](https://docs.python.org/3/library/threading.html) module according to Pypeln's general [architecture](https://cgarciae.github.io/pypeln/advanced/#architecture). Use this module when you are in need to perform some synchronous IO operations and DONT need to perform heavy CPU operations.

Most functions in this module return a `pl.thread.Stage` object which implement the `Iterable` interface which enables you to combine it seamlessly with regular Python code.


### Iterable
You can iterate over any `p.thread.Stage` to get back the results of its computation:

```python
import pypeln as pl
import time
from random import random

def slow_add1(x):
    time.sleep(random()) # <= some slow computation
    return x + 1

def slow_gt3(x):
    time.sleep(random()) # <= some slow computation
    return x > 3

data = range(10) # [0, 1, 2, ..., 9] 

stage = pl.thread.map(slow_add1, data, workers=3, maxsize=4)
stage = pl.thread.filter(slow_gt3, stage, workers=2)

for x in stage:
    print(x) # e.g. 5, 6, 9, 4, 8, 10, 7
```
At each stage the you can specify the numbers of `workers`. The `maxsize` parameter limits the maximum amount of elements that the stage can hold simultaneously.

### Runtime worker adjustment

Stages created with `map`, `filter`, `flat_map` (and `each`) expose a `stage.controller` (`pl.thread.WorkerController`) handle that lets you change the number of workers of a running pipeline without re-reading the source or restarting the chain:

```python
import pypeln as pl
import threading

stage = pl.thread.map(slow_add1, data, workers=2)
stage = pl.thread.filter(slow_gt3, stage, workers=1)

results = []
t = threading.Thread(target=lambda: results.extend(stage), daemon=True)
t.start()

stage.controller.wait_running(timeout=5)  # wait until the pipeline is running
stage.controller.set_workers(4)           # scale the filter stage up
stage.controller.set_workers(1)           # and back down (must stay positive)

t.join()
```

New workers build their own resources (`on_start` runs on their own thread); workers selected for removal stop pulling new elements, deliver their current element (including ongoing `flat_map` expansions), run `on_done`, and exit normally. After the pipeline ends or `controller.close()` is called, further adjustments are rejected.
"""

from .api.concat import concat
from .api.each import each
from .api.filter import filter
from .api.flat_map import flat_map
from .api.from_iterable import from_iterable
from .api.map import map
from .api.ordered import ordered
from .api.run import run
from .api.to_iterable import to_iterable
from .controller import WorkerController
from .queue import IterableQueue, OutputQueues
from .supervisor import Supervisor
from .utils import Namespace
from .worker import StageParams, Worker, WorkerInfo, start_workers
