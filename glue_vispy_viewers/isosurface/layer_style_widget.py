import os

from qtpy import QtWidgets

from glue_qt.utils import load_ui
from echo.qt import autoconnect_callbacks_to_qt


class IsosurfaceLayerStyleWidget(QtWidgets.QWidget):

    def __init__(self, layer_artist):

        super(IsosurfaceLayerStyleWidget, self).__init__()

        self.ui = load_ui('layer_style_widget.ui', self,
                          directory=os.path.dirname(__file__))

        self.state = layer_artist.state

        self.layer_artist = layer_artist
        self.layer = layer_artist.layer

        connect_kwargs = {'value_alpha': dict(value_range=(0., 1.)),
                          'value_step': dict(value_range=(1, 10))}
        self._connections = autoconnect_callbacks_to_qt(self.state, self.ui, connect_kwargs)
