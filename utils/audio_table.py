from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

from PyQt5.QtCore import Qt, QSortFilterProxyModel, pyqtSignal as Signal
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import (
    QDockWidget,
    QTableView,
    QWidget,
    QVBoxLayout,
    QLineEdit,
    QTabWidget,
    QCheckBox,
    QAbstractItemView,
)

from .session_model import SessionModel
from .selection_bus import SelectionBus


class _AudioTable(QWidget):
    """Widget showing a filterable table of audio files."""

    def __init__(self, session: SessionModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._session = session

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.filter_edit = QLineEdit(self)
        self.filter_edit.setPlaceholderText("Filter…")
        layout.addWidget(self.filter_edit)

        self.model = QStandardItemModel(0, 3, self)
        self.model.setHorizontalHeaderLabels(["File Name", "File Path", "Similarity"])

        self.proxy = QSortFilterProxyModel(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.proxy.setFilterKeyColumn(-1)

        self.view = QTableView(self)
        self.view.setModel(self.proxy)
        self.view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.view.horizontalHeader().setStretchLastSection(True)
        self.view.setSortingEnabled(True)        
        layout.addWidget(self.view)

        self.filter_edit.textChanged.connect(self.proxy.setFilterFixedString)

    def set_indices(self, idxs: List[int]) -> None:
        """Populate the table with the given session indices."""
        self.model.setRowCount(0)
        paths = self._session.im_list
        for idx in idxs:
            if 0 <= idx < len(paths):
                p = paths[idx]
                items = [
                    QStandardItem(Path(p).name),
                    QStandardItem(p),
                    QStandardItem("0.0"),  # placeholder for similarity
                ]
                self.model.appendRow(items)

    def set_similarity(self, pairs: Sequence[tuple[int, float]]) -> None:
        """Populate the table with indices and similarity scores."""
        self.model.setRowCount(0)
        paths = self._session.im_list
        for idx, score in pairs:
            if 0 <= idx < len(paths):
                path = paths[idx]
                name_item = QStandardItem(Path(path).name)
                path_item = QStandardItem(path)
                sim_item = QStandardItem(f"{score:.3f}")
                sim_item.setData(float(score), Qt.EditRole)
                self.model.appendRow([name_item, path_item, sim_item])
        self.view.sortByColumn(2, Qt.DescendingOrder)

class AudioTableDock(QDockWidget):
    """Dock widget containing filterable tables of audio files for each model."""

    historyChanged = Signal()
    labelDoubleClicked = Signal(str)

    def __init__(self, bus: SelectionBus, parent: QWidget | None = None) -> None:
        super().__init__("Audio Files", parent)
        self.bus = bus

        self.tabs = QTabWidget(self)
        self.setWidget(self.tabs)

        # Exposed checkboxes to keep interface compatibility
        self.hide_selected_cb = QCheckBox("Hide selected")
        self.hide_modified_cb = QCheckBox("Hide modified")

        self._history: List[List[int]] = []
        self._hist_pos = -1

        self._sessions: dict[str, SessionModel] = {}

        self.tabs.currentChanged.connect(self._on_tab_changed)

    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------
    @property
    def view(self) -> QTableView:
        widget = self.tabs.currentWidget()
        return widget.view if isinstance(widget, _AudioTable) else QTableView()

    def set_model(self, session: SessionModel, name: str = "Model") -> None:
        """Clear existing tabs and show the given session as single tab."""
        self.clear()
        self.add_model(name, session)

    def add_model(self, name: str, session: SessionModel) -> None:
        table = _AudioTable(session, self)
        self.tabs.addTab(table, name)
        self._sessions[name] = session
        # show all files initially
        table.set_indices(list(range(len(session.im_list))))
        self.tabs.setCurrentWidget(table)

    def show_similarity_results(
        self,
        ref_name: str,
        results: Sequence[tuple[int, float]],
        session: SessionModel,
    ) -> None:
        """Add or update a tab containing similarity-ranked songs."""
        tab_title = f"Similarity – {ref_name}"
        table: _AudioTable | None = None

        for idx in range(self.tabs.count()):
            if self.tabs.tabText(idx) == tab_title:
                widget = self.tabs.widget(idx)
                if isinstance(widget, _AudioTable):
                    table = widget
                else:
                    table = _AudioTable(session, self)
                    self.tabs.removeTab(idx)
                    self.tabs.insertTab(idx, table, tab_title)
                break

        if table is None:
            table = _AudioTable(session, self)
            self.tabs.addTab(table, tab_title)

        table.set_similarity(results)
        self._sessions[tab_title] = session
        self.tabs.setCurrentWidget(table)

        idxs = [idx for idx, _ in results]
        if idxs:
            self._history = [idxs]
            self._hist_pos = 0
        else:
            self._history = []
            self._hist_pos = -1
        self.historyChanged.emit()

    def clear(self) -> None:
        self.tabs.clear()
        self._sessions.clear()
        self._history.clear()
        self._hist_pos = -1

    # Methods used by existing code -----------------------------------
    def update_images(self, idxs: List[int], **kwargs) -> None:  # type: ignore[override]
        widget = self.tabs.currentWidget()
        if isinstance(widget, _AudioTable):
            widget.set_indices(idxs)
            self._history = self._history[: self._hist_pos + 1]
            self._history.append(idxs)
            self._hist_pos += 1
            self.historyChanged.emit()

    def show_overview(self, triplets, session: SessionModel) -> None:
        idxs: List[int] = []
        for a, b, c in triplets:
            for v in (a, b, c):
                if v is not None:
                    idxs.append(v)
        self.update_images(idxs, sort=False)

    def set_use_full_images(self, flag: bool) -> None:  # compatibility stub
        pass

    def go_back(self) -> None:
        if self._hist_pos > 0:
            self._hist_pos -= 1
            idxs = self._history[self._hist_pos]
            widget = self.tabs.currentWidget()
            if isinstance(widget, _AudioTable):
                widget.set_indices(idxs)
            self.historyChanged.emit()

    def go_forward(self) -> None:
        if self._hist_pos + 1 < len(self._history):
            self._hist_pos += 1
            idxs = self._history[self._hist_pos]
            widget = self.tabs.currentWidget()
            if isinstance(widget, _AudioTable):
                widget.set_indices(idxs)
            self.historyChanged.emit()

    def can_go_back(self) -> bool:
        return self._hist_pos > 0

    def can_go_forward(self) -> bool:
        return self._hist_pos + 1 < len(self._history)

    def _on_tab_changed(self, index: int) -> None:
        # Reset history when switching tabs
        self._history.clear()
        self._hist_pos = -1