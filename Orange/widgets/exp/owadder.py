from Orange.widgets.widget import OWWidget, Input, Output


class Adder(OWWidget):
    name = "Add two integers"
    description = "Add two numbers"
    icon = "icons/add.svg"

    class Inputs:
        a = Input("A", int)
        b = Input("B", int)

    class Outputs:
        sum = Output("A + B", int)

    want_main_area = False

    def __init__(self):
        super().__init__()
        self.a = None
        self.b = None

    @Inputs.a
    def set_A(self, a):
        """Set input 'A'."""
        self.a = a

    @Inputs.b
    def set_B(self, b):
        """Set input 'B'."""
        self.b = b

    def handleNewSignals(self):
        """Reimplemeted from OWWidget."""
        if self.a is not None and self.b is not None:
            self.Outputs.sum.send(self.a + self.b)
        else:
            # Clear the channel by sending `None`
            self.Outputs.sum.send(None)
