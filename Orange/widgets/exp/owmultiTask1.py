from pathlib import Path
import numpy
from functools import reduce, partial
from Orange.widgets.widget import OWWidget
from Orange.widgets import gui
from Orange.widgets.utils.signals import Input

from fastai.vision import ImageDataBunch, nn, Flatten, Learner, accuracy

import concurrent.futures
from Orange.widgets.utils.concurrent import ThreadExecutor, FutureWatcher, methodinvoke
from AnyQt.QtCore import QThread, pyqtSlot

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

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


class PNNM(OWWidget):
    name = "Pytorch CNN"
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

        # Device configuration
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Hyper parameters
        num_epochs = 5
        num_classes = 10
        batch_size = 100
        learning_rate = 0.001

        dir_path = Path(__file__).resolve()
        data_path = f'{dir_path.parent.parent.parent}/datasets/'

        # MNIST dataset
        self.train_dataset = torchvision.datasets.MNIST(root=data_path,
                                                   train=True,
                                                   transform=transforms.ToTensor(),
                                                   download=False)

        self.test_dataset = torchvision.datasets.MNIST(root=data_path,
                                                  train=False,
                                                  transform=transforms.ToTensor())

        # Data loader
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False)

        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)

        # self.model = ConvNet(num_classes).to(self.device)
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
        ).to(self.device)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

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

        fit_model = partial(train_model, self.model, 5, self.train_loader, self.test_loader, self.device,
                            self.criterion, self.optimizer, callback=callback)

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

        self.model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

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


def train_model(
    model, epochs, train_loader, test_loader, device, criterion, optimizer,
        proportions=None, random_state=None, callback=None):

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

    total_step = len(train_loader)
    print(total_step)
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, epochs, i + 1, total_step, loss.item()))

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
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
    return 100 * correct / total
