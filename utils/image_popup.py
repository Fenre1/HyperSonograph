from __future__ import annotations

from pathlib import Path
from typing import Mapping, Any

from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QLineEdit,
    QLabel,
)
from PyQt5.QtCore import Qt


class ImageMetadataDialog(QDialog):
    """Dialog displaying metadata for a single image.

    The previous version of this dialog displayed the image itself and allowed
    the user to select a region to perform similarity queries on.  This simplified
    version only shows the filename and associated metadata.
    """

    def __init__(
        self,
        session,
        idx: int,
        parent=None,
        indices: list[int] | None = None,
    ):
        super().__init__(parent)

        self._session = session
        self._indices = indices or list(range(len(session.im_list)))
        self._pos = self._indices.index(idx) if idx in self._indices else 0
        self._image_path = ""

        self.file_label = QLabel()
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter...")
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Key", "Value"])
        self.table.horizontalHeader().setStretchLastSection(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.file_label)
        layout.addWidget(self.filter_edit)
        layout.addWidget(self.table)

        self.filter_edit.textChanged.connect(self._apply_filter)

        self._load_image(self._indices[self._pos])

    def _populate_table(self, metadata: Mapping[str, Any]):
        self.table.setRowCount(0)
        for key, value in metadata.items():
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(str(key)))
            self.table.setItem(r, 1, QTableWidgetItem(str(value)))

    def _apply_filter(self, text: str):
        text = text.lower()
        for row in range(self.table.rowCount()):
            key_item = self.table.item(row, 0)
            val_item = self.table.item(row, 1)
            combined = f"{key_item.text()} {val_item.text()}".lower()
            match = text in combined
            self.table.setRowHidden(row, not match)

    def _load_image(self, idx: int) -> None:
        self._image_path = self._session.im_list[idx]
        fname = Path(self._image_path).name
        self.setWindowTitle(fname)
        self.file_label.setText(fname)
        row = self._session.metadata[
            self._session.metadata["image_path"] == self._image_path
        ]
        meta = row.iloc[0].to_dict() if not row.empty else {}
        self._populate_table(meta)
        self.filter_edit.clear()

    def _show_prev(self) -> None:
        if self._pos > 0:
            self._pos -= 1
            self._load_image(self._indices[self._pos])

    def _show_next(self) -> None:
        if self._pos + 1 < len(self._indices):
            self._pos += 1
            self._load_image(self._indices[self._pos])

    def keyPressEvent(self, event):  # noqa: N802 - Qt naming convention
        if event.key() == Qt.Key_Left:
            self._show_prev()
            event.accept()
        elif event.key() == Qt.Key_Right:
            self._show_next()
            event.accept()
        else:
            super().keyPressEvent(event)


_open_dialogs: list[ImageMetadataDialog] = []


def show_image_metadata(session, idx: int, parent=None):
    """Display the metadata dialog for the given image index."""
    indices = getattr(parent, "_current_indices", None)
    dlg = ImageMetadataDialog(session, idx, parent=parent, indices=indices)
    _open_dialogs.append(dlg)
    dlg.finished.connect(lambda *_: _open_dialogs.remove(dlg))
    dlg.show()
