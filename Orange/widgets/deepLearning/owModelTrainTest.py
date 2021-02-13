from functools import partial
from Orange.widgets.widget import OWWidget
from Orange.widgets import gui
from Orange.widgets.utils.signals import Input, Output
from Orange.data import Table, Domain, ContinuousVariable


import concurrent.futures
from Orange.widgets.utils.concurrent import ThreadExecutor, FutureWatcher, methodinvoke
from AnyQt.QtCore import QThread, pyqtSlot

import torch
import torch.nn as nn
from torch.utils.data import DataLoader



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


class ModelTrainTest(OWWidget):
    name = "模型训练与测试(train & test)"
    description = "训练深度学习模型"
    icon = "icons/train.png"
    keywords = ['xunlianheceshi', 'xunlianyuceshi', 'ceshi', 'moxing']
    category = 'deeplearning'

    want_main_area = True

    learning_rate = 0.001
    epochs = 5

    class Inputs:
        train_data = Input('训练数据(Train Data)', DataLoader, default=True, replaces=['Data'])
        test_data = Input('测试数据(Test Data)', DataLoader, replaces=['Data'])
        model = Input('模型(Model)', nn.Module, replaces=['Model'])

    class Outputs:
        losses = Output('损失函数值(Loss)', Table, default=True, replaces=['Loss'])
        model = Output('模型(Model)', nn.Module, replaces=['Model'])

    def __init__(self):
        super().__init__()
        self.train_data = None
        self.test_data = None
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.index = 0

        self.output_results = []   # [[id1, loss1], [id2, loss2],...]
        attrs = [ContinuousVariable('id'), ContinuousVariable('损失函数值')]
        self.domain = Domain(attrs)

        info_box = gui.widgetBox(self.controlArea, "输入信息:")
        self.info_label_train = gui.widgetLabel(info_box, '训练数据: ?')
        self.info_label_test = gui.widgetLabel(info_box, '测试数据: ?')
        self.info_label_model = gui.widgetLabel(info_box, '模型: ?')

        training_box = gui.widgetBox(self.controlArea, '训练参数')
        gui.doubleSpin(
            training_box,
            self,
            "learning_rate",
            minv=0,
            maxv=0.05,
            step=0.0002,
            label="学习率:",
        )
        gui.spin(
            training_box,
            self,
            "epochs",
            minv=1,
            maxv=10,
            step=1,
            label="训练周期:",
        )

        self.train_button = gui.button(self.controlArea, self, "开始训练", callback=self.start)
        self.label = gui.label(self.mainArea, self, "训练结果")

        #: The current evaluating task (if any)
        self._task = None  # type: Optional[Task]
        #: An executor we use to submit learner evaluations into a thread pool
        self._executor = ThreadExecutor()

        # Device configuration
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Hyper parameters

        # self.model = ConvNet(num_classes).to(self.device)
        # Loss and optimizer

    @Inputs.train_data
    def set_train_data(self, data):
        """Set the input number."""
        self.train_data = data
        if self.train_data is None:
            self.info_label_train.setText("必须有训练数据")
        else:
            self.info_label_train.setText(f'训练数据一共有 {len(data)} 个 batch')

    @Inputs.test_data
    def set_test_data(self, data):
        """Set the input number."""
        self.test_data = data
        if self.test_data is None:
            self.info_label_test.setText("必须有测试数据")
        else:
            self.info_label_test.setText(f'测试数据一共有 {len(data)} 个 batch')

    @Inputs.model
    def set_model(self, model):
        """Set the input number."""
        self.model = model
        if self.model is None:
            self.info_label_model.setText("必须有模型")
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.info_label_model.setText('')

    # def handleNewSignals(self):
    def start(self):
        self._update()

    def _update(self):
        if self._task is not None:
            # First make sure any pending tasks are cancelled.
            self.cancel()
        assert self._task is None

        if self.train_data is None and self.test_data is None:
            return
        # collect all learners for which results have not yet been computed
        if self.model is None:
            return

        # setup the task state
        self._task = task = Task()
        # The learning_curve[_with_test_data] also takes a callback function
        # to report the progress. We instrument this callback to both invoke
        # the appropriate slots on this widget for reporting the progress
        # (in a thread safe manner) and to implement cooperative cancellation.
        set_progress = methodinvoke(self, "setProgressValue", (float,))

        def callback(finished, loss):
            # check if the task has been cancelled and raise an exception
            # from within. This 'strategy' can only be used with code that
            # properly cleans up after itself in the case of an exception
            # (does not leave any global locks, opened file descriptors, ...)
            if task.cancelled:
                raise KeyboardInterrupt()
            set_progress(finished * 100)
            self.output_results.append([self.index, loss])
            self.index += 1
            # self.Outputs.losses.send(Table.from_list(self.domain, self.output_results))

        self.progressBarInit()
        # Submit the evaluation function to the executor and fill in the
        # task with the resultant Future.
        # task.future = self._executor.submit(self.learn.fit_one_cycle(1))

        fit_model = partial(train_model, self.model, self.epochs, self.train_data, self.test_data, self.device,
                            self.criterion, self.optimizer, self.train_button, callback=callback)

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
        self.train_button.setEnabled(True)

        self.model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.test_data:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            self.label.setText('测试数据集的准确率为: {} %'.format(100 * correct / total))
        self.Outputs.model.send(self.model)
        self.Outputs.losses.send(Table.from_list(self.domain, self.output_results))


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


def train_model(
    model, epochs, train_loader, test_loader, device, criterion, optimizer,
        train_button, callback=None):

    update_index = callback
    train_button.setEnabled(False)

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

            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, epochs, i + 1, total_step, loss.item()))
                progress = 1.0 / epochs * i / total_step + float(epoch) / epochs
                print(f'{progress * 100:.0f}%')
                update_index(progress, loss.item())

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
