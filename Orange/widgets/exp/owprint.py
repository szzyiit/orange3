from Orange.widgets.widget import OWWidget, Input
from Orange.widgets import gui


class Print(OWWidget):
    name = "Print"
    description = "Print out a number"
    icon = "icons/print.svg"

    class Inputs:
        number = Input("Number", int)

    want_main_area = False

    def __init__(self):
        super().__init__()
        self.number = None
        self.label = gui.widgetLabel(self.controlArea, "The number is: ??")

    @Inputs.number
    def set_number(self, number):
        """Set the input number."""
        self.number = number
        if self.number is None:
            self.label.setText("The number is: ??")
        else:
            self.label.setText("The number is {}".format(self.number))
