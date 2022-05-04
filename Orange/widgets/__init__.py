"""

"""
import os
import sysconfig

import pkg_resources

from orangecanvas.registry import CategoryDescription
from orangecanvas.registry.utils import category_from_package_globals
import orangewidget.workflow.discovery


# Entry point for main Orange categories/widgets discovery
def widget_discovery(discovery):
    # type: (orangewidget.workflow.discovery.WidgetDiscovery) -> None
    dist = pkg_resources.get_distribution("Orange3-zh")
    pkgs = [
        "Orange.widgets.data",
        "Orange.widgets.visualize",
        "Orange.widgets.model",
        "Orange.widgets.evaluate",
        "Orange.widgets.unsupervised",
        "Orange.widgets.reinforcement",
        "Orange.widgets.deepLearning",
    ]
    for pkg in pkgs:
        discovery.handle_category(category_from_package_globals(pkg))
    # manually described category (without 'package' definition)
    # discovery.handle_category(
    #     CategoryDescription(
    #         name="变换(Transform)",
    #         priority=1,
    #         background="#FF9D5E",
    #         icon="data/icons/Transform.svg",
    #         package=__package__,
    #     )
    # )
    for pkg in pkgs:
        discovery.process_category_package(pkg, distribution=dist)


WIDGET_HELP_PATH = (
    ("{DEVELOP_ROOT}/doc/visual-programming/build/htmlhelp/index.html", None),
    (os.path.join(sysconfig.get_path("data"),
                  "share/help/en/orange3/htmlhelp/index.html"),
     None),
    ("https://docs.biolab.si/orange/3/visual-programming/", ""),
)
