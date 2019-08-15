import logging
import numpy
from functools import reduce, partial
from Orange.widgets.widget import OWWidget
from Orange.widgets import gui
from Orange.widgets.utils.signals import Input

from fastai.vision import ImageDataBunch, nn, Flatten, Learner, accuracy

import concurrent.futures
from Orange.widgets.utils.concurrent import ThreadExecutor, FutureWatcher, methodinvoke
from Orange.widgets.utils.widgetpreview import WidgetPreview
from AnyQt.QtCore import QThread, pyqtSlot

import fastai
import fastprogress
from fastai.basic_train import *
from fastai.core import *
class progress_disabled_ctx():
    "Context manager to disable the progress update bar and Recorder print."
    def __init__(self,learn:Learner):
        self.learn = learn

    def __enter__(self):
        #silence progress bar
        fastprogress.fastprogress.NO_BAR = True
        fastai.basic_train.master_bar,fastai.basic_train.progress_bar = fastprogress.force_console_behavior()
        self.orig_callback_fns = copy(self.learn.callback_fns)
        rec_name = [x for x in self.learn.callback_fns if hasattr(x, 'func') and x.func == Recorder]
        if len(rec_name):
            rec_idx = self.learn.callback_fns.index(rec_name[0])
            self.learn.callback_fns[rec_idx] = partial(Recorder, add_time=False, silent=True) #silence recorder
        return self.learn

    def __exit__(self, *args):
        fastai.basic_train.master_bar,fastai.basic_train.progress_bar = master_bar,progress_bar
        self.learn.callback_fns = self.orig_callback_fns

class Task:
    """
    A class that will hold the state for an learner evaluation.
    """

    #: A concurrent.futures.Future with our (eventual) results.
    #: The OWLearningCurveC class must fill this field
    future = ...  # type: concurrent.futures.Future

    #: FutureWatcher. Likewise this will be filled by OWLearningCurveC
    watcher = ...  # type: FutureWatcher

    #: True if this evaluation has been cancelled. The OWLearningCurveC
    #: will setup the task execution environment in such a way that this
    #: field will be checked periodically in the worker thread and cancel
    #: the computation if so required. In a sense this is the only
    #: communication channel in the direction from the OWLearningCurve to the
    #: worker thread
    cancelled = False  # type: bool

    def cancel(self):
        """
        Cancel the task.

        Set the `cancelled` field to True and block until the future is done.
        """
        # set cancelled state
        self.cancelled = True
        # cancel the future. Note this succeeds only if the execution has
        # not yet started (see `concurrent.futures.Future.cancel`) ..
        self.future.cancel()
        # ... and wait until computation finishes
        concurrent.futures.wait([self.future])


class CNNM(OWWidget):
    name = "M CNN"
    description = ""
    # icon = "icons/robot.svg"

    want_main_area = True

    class Inputs:
        data = Input('Data', ImageDataBunch, default=True)

    def __init__(self):
        super().__init__()
        self.learn = None

        # train_button = gui.button(self.controlArea, self, "开始训练", callback=self.train)
        self.label = gui.label(self.mainArea, self, "模型结构")

        #: The current evaluating task (if any)
        self._task = None  # type: Optional[Task]
        #: An executor we use to submit learner evaluations into a thread pool
        self._executor = ThreadExecutor()

        self.model = nn.Sequential(
            self.conv(1, 8),  # 14
            nn.BatchNorm2d(8),
            nn.ReLU(),
            self.conv(8, 16),  # 7
            nn.BatchNorm2d(16),
            nn.ReLU(),
            self.conv(16, 32),  # 4
            nn.BatchNorm2d(32),
            nn.ReLU(),
            self.conv(32, 16),  # 2
            nn.BatchNorm2d(16),
            nn.ReLU(),
            self.conv(16, 10),  # 1
            nn.BatchNorm2d(10),
            Flatten()  # remove (1,1) grid
        )

    def handleNewSignals(self):
        self._update()

    def _update(self):
        if self._task is not None:
            # First make sure any pending tasks are cancelled.
            self.cancel()
        assert self._task is None

        if self.data is None:
            return
        # collect all learners for which results have not yet been computed
        if not self.learn:
            return

        # setup the task state
        self._task = task = Task()
        # The learning_curve[_with_test_data] also takes a callback function
        # to report the progress. We instrument this callback to both invoke
        # the appropriate slots on this widget for reporting the progress
        # (in a thread safe manner) and to implement cooperative cancellation.
        set_progress = methodinvoke(self, "setProgressValue", (float,))

        def callback(finished):
            # check if the task has been cancelled and raise an exception
            # from within. This 'strategy' can only be used with code that
            # properly cleans up after itself in the case of an exception
            # (does not leave any global locks, opened file descriptors, ...)
            if task.cancelled:
                raise KeyboardInterrupt()
            set_progress(finished * 100)

        self.progressBarInit()
        # Submit the evaluation function to the executor and fill in the
        # task with the resultant Future.
        # task.future = self._executor.submit(self.learn.fit_one_cycle(1))

        with progress_disabled_ctx(self.learn) as learn:
            fit_model = partial(my_fit, learn, 1, callback=callback)
            task.future = self._executor.submit(fit_model)
            # Setup the FutureWatcher to notify us of completion
            task.watcher = FutureWatcher(task.future)
            # by using FutureWatcher we ensure `_task_finished` slot will be
            # called from the main GUI thread by the Qt's event loop
            task.watcher.done.connect(self._task_finished)

    @pyqtSlot(float)
    def setProgressValue(self, value):
        assert self.thread() is QThread.currentThread()
        self.progressBarSet(value)

    @pyqtSlot(concurrent.futures.Future)
    def _task_finished(self, f):
        """
        Parameters
        ----------
        f : Future
            The future instance holding the result of learner evaluation.
        """
        assert self.thread() is QThread.currentThread()
        assert self._task is not None
        assert self._task.future is f
        assert f.done()

        self._task = None
        self.progressBarFinished()

        # try:
        #     result = f.result()  # type: List[Results]
        # except Exception as ex:
        #     # Log the exception with a traceback
        #     log = logging.getLogger()
        #     log.exception(__name__, exc_info=True)
        #     self.error("Exception occurred during evaluation: {!r}".format(ex))
        #     # clear all results
        #     self.result= None
        # else:
        print(self.learn.validate())
            # ... and update self.results

    def cancel(self):
        """
        Cancel the current task (if any).
        """
        if self._task is not None:
            self._task.cancel()
            assert self._task.future.done()
            # disconnect the `_task_finished` slot
            self._task.watcher.done.disconnect(self._task_finished)
            self._task = None

    def onDeleteWidget(self):
        self.cancel()
        super().onDeleteWidget()

    def conv(self, ni, nf):
        return nn.Conv2d(ni, nf, kernel_size=3, stride=2, padding=1)

    def train(self):
        if self.learn is None:
            return
        self.learn.fit_one_cycle(3)

    @Inputs.data
    def set_data(self, data):
        if data is not None:
            self.data = data
            self.learn = Learner(self.data, self.model, loss_func=nn.CrossEntropyLoss(), metrics=accuracy,
                                 add_time=False, bn_wd=False, silent=True)
            self.label.setText(self.learn.summary())
        else:
            self.data = None


def my_fit(
    learn, epochs, proportions=None, random_state=None, callback=None
):

    if proportions is None:
        proportions = numpy.linspace(0.0, 1.0, 10 + 1, endpoint=True)[1:]

    def select_proportion_preproc(data, p, rstate=None):
        assert 0 < p <= 1
        rstate = numpy.random.RandomState(None) if rstate is None else rstate
        indices = rstate.permutation(len(data))
        n = int(numpy.ceil(len(data) * p))
        return data[indices[:n]]

    if callback is not None:
        parts_count = len(proportions)
        callback_wrapped = lambda part: lambda value: callback(
            value / parts_count + part / parts_count
        )
    else:
        callback_wrapped = lambda part: None

    learn.fit_one_cycle(epochs)

    # results = [
    #     Orange.evaluation.CrossValidation(
    #         data,
    #         learners,
    #         k=folds,
    #         preprocessor=lambda data, p=p: select_proportion_preproc(data, p),
    #         callback=callback_wrapped(i),
    #     )
    #     for i, p in enumerate(proportions)
    # ]
    # print(learn.validate())
    return learn.validate()
