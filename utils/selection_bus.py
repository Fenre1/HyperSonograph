from PyQt5.QtCore import QObject, pyqtSignal as Signal

class SelectionBus(QObject):
    """Send the current selection of hyperedges and images to all views."""

    edgesChanged = Signal(list)   # selected hyperedge names
    imagesChanged = Signal(list)  # selected image indices

    def set_edges(self, names: list[str]):
        self.edgesChanged.emit(names)

    def set_images(self, idxs: list[int]):
        self.imagesChanged.emit(idxs)