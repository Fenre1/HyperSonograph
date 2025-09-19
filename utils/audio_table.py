from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from PyQt5.QtCore import Qt, QSortFilterProxyModel, pyqtSignal as Signal
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDockWidget,
    QLineEdit,
    QStyle,
    QTabBar,
    QTabWidget,
    QTableView,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


from .session_model import SegmentMatchResult, SessionModel
from .selection_bus import SelectionBus
INDEX_ROLE = Qt.UserRole + 1

class TagFilterProxyModel(QSortFilterProxyModel):
    """Proxy model that filters rows based on assigned song tags."""

    def __init__(self, table: "_AudioTable", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._table = table
        self._allowed_tags: set[str] = set()
        self._include_untagged = True

    def set_tag_filters(self, tags: Iterable[str], include_untagged: bool) -> None:
        self._allowed_tags = {str(tag) for tag in tags}
        self._include_untagged = bool(include_untagged)
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent) -> bool:  # type: ignore[override]
        if not super().filterAcceptsRow(source_row, source_parent):
            return False

        session = self._table.session
        if session is None:
            return True

        model = self.sourceModel()
        if not isinstance(model, QStandardItemModel):
            return True

        item = model.item(source_row, 0)
        if item is None:
            return True

        value = item.data(INDEX_ROLE)
        if value is None:
            return True

        try:
            song_idx = int(value)
        except (TypeError, ValueError):
            return True

        tags = session.get_tags_for_song(song_idx)
        if not tags:
            return self._include_untagged
        return bool(tags & self._allowed_tags)

class _AudioTableTabBar(QTabBar):
    """Tab bar supporting middle-click closing for closable tabs."""

    middleClickCloseRequested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._closable_checker: Callable[[int], bool] | None = None

    def set_closable_checker(self, checker: Callable[[int], bool]) -> None:
        self._closable_checker = checker

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MiddleButton:
            index = self.tabAt(event.pos())
            if index >= 0:
                closable = True
                if self._closable_checker is not None:
                    closable = self._closable_checker(index)
                if closable:
                    self.middleClickCloseRequested.emit(index)
                    event.accept()
                    return
        super().mouseReleaseEvent(event)

class _AudioTable(QWidget):
    """Widget showing a filterable table of audio files."""

    selectionChanged = Signal(list)

    def __init__(self, session: SessionModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._session = session
        self._is_similarity_tab = False
        self._similarity_ref_index: int | None = None
        self._similarity_ref_name: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.filter_edit = QLineEdit(self)
        self.filter_edit.setPlaceholderText("Filter…")
        layout.addWidget(self.filter_edit)

        self.model = QStandardItemModel(0, 4, self)
        self.model.setHorizontalHeaderLabels(["File Name", "File Path", "Similarity", "Tags"])

        self.proxy = TagFilterProxyModel(self, self)
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
        self.view.selectionModel().selectionChanged.connect(self._on_selection_changed)
        self._allowed_tags: set[str] = set()
        self._include_untagged = True

    @property
    def session(self) -> SessionModel:
        return self._session

    def set_indices(self, idxs: List[int]) -> None:
        """Populate the table with the given session indices."""
        self.model.setRowCount(0)
        self.view.clearSelection()
        self._is_similarity_tab = False
        self._similarity_ref_index = None
        self._similarity_ref_name = None
        paths = self._session.im_list
        for idx in idxs:
            if 0 <= idx < len(paths):
                p = paths[idx]
                items = [
                    QStandardItem(Path(p).name),
                    QStandardItem(p),
                    QStandardItem(""),
                    QStandardItem(self._format_tags(idx)),
                ]
                for item in items:
                    item.setEditable(False)
                    item.setData(int(idx), INDEX_ROLE)
                self.model.appendRow(items)

    def set_similarity(
        self,
        pairs: Sequence[tuple[int, float]],
        *,
        ref_index: int | None = None,
        ref_name: str | None = None,
    ) -> None:
        """Populate the table with indices and similarity scores."""
        self.model.setRowCount(0)
        self.view.clearSelection()
        self._is_similarity_tab = True
        self._similarity_ref_index = ref_index
        self._similarity_ref_name = ref_name
        paths = self._session.im_list
        for idx, score in pairs:
            if 0 <= idx < len(paths):
                path = paths[idx]
                name_item = QStandardItem(Path(path).name)
                path_item = QStandardItem(path)
                sim_item = QStandardItem(f"{score:.3f}")
                sim_item.setData(float(score), Qt.EditRole)
                for item in (name_item, path_item, sim_item):
                    item.setEditable(False)
                    item.setData(int(idx), INDEX_ROLE)
                tags_item = QStandardItem(self._format_tags(idx))
                tags_item.setEditable(False)
                tags_item.setData(int(idx), INDEX_ROLE)
                self.model.appendRow([name_item, path_item, sim_item, tags_item])
        self.view.sortByColumn(2, Qt.DescendingOrder)

    def is_similarity_tab(self) -> bool:
        return self._is_similarity_tab

    def similarity_reference_index(self) -> int | None:
        return self._similarity_ref_index

    def similarity_reference_name(self) -> Optional[str]:
        return self._similarity_ref_name

    def selected_song_indices(self) -> List[int]:
        selection = self.view.selectionModel()
        if selection is None:
            return []
        idxs: list[int] = []
        for proxy_idx in selection.selectedRows():
            src = self.proxy.mapToSource(proxy_idx)
            if not src.isValid():
                continue
            item = self.model.item(src.row(), 0)
            if not item:
                continue
            value = item.data(INDEX_ROLE)
            if isinstance(value, (int, float)):
                idxs.append(int(value))
        return idxs

    def best_segment_match(self, other_song_idx: int) -> SegmentMatchResult | None:
        if not self._is_similarity_tab:
            return None
        if self._similarity_ref_index is None:
            return None
        if not self._session.has_segment_features():
            return None
        return self._session.best_matching_segment_pair(
            self._similarity_ref_index,
            int(other_song_idx),
        )

    def _on_selection_changed(self, *_args) -> None:
        self.selectionChanged.emit(self.selected_song_indices())

    def set_tag_filter(self, allowed: Iterable[str], include_untagged: bool) -> None:
        self._allowed_tags = {str(tag) for tag in allowed}
        self._include_untagged = bool(include_untagged)
        self.proxy.set_tag_filters(self._allowed_tags, self._include_untagged)

    def refresh_tags_for_indices(self, indices: Sequence[int]) -> None:
        if not indices:
            return
        idxs = {int(i) for i in indices}
        for row in range(self.model.rowCount()):
            item = self.model.item(row, 0)
            if not item:
                continue
            value = item.data(INDEX_ROLE)
            if not isinstance(value, (int, float)):
                continue
            song_idx = int(value)
            if song_idx not in idxs:
                continue
            tags_item = self.model.item(row, 3)
            if tags_item is not None:
                text = self._format_tags(song_idx)
                tags_item.setText(text)
        self.proxy.invalidateFilter()

    def _format_tags(self, song_idx: int) -> str:
        tags = sorted(self._session.get_tags_for_song(song_idx)) if self._session else []
        return ", ".join(tags)

class AudioTableDock(QDockWidget):
    """Dock widget containing filterable tables of audio files for each model."""

    historyChanged = Signal()
    labelDoubleClicked = Signal(str)
    similarityPairSelected = Signal(object)

    def __init__(self, bus: SelectionBus, parent: QWidget | None = None) -> None:
        super().__init__("Audio Files", parent)
        self.bus = bus

        self.tabs = QTabWidget(self)
        self._closable_tabs: dict[QWidget, bool] = {}
        tab_bar = _AudioTableTabBar(self.tabs)
        tab_bar.set_closable_checker(self._is_tab_index_closable)
        tab_bar.middleClickCloseRequested.connect(self._close_tab)
        self.tabs.setTabBar(tab_bar)
        self.setWidget(self.tabs)

        # Exposed checkboxes to keep interface compatibility
        self.hide_selected_cb = QCheckBox("Hide selected")
        self.hide_modified_cb = QCheckBox("Hide modified")

        self._history: List[List[int]] = []
        self._hist_pos = -1

        self._sessions: dict[str, SessionModel] = {}
        self._active_tags: set[str] = set()
        self._include_untagged = True

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
        self._configure_table(table)
        self._sessions[name] = session
        # show all files initially
        table.set_indices(list(range(len(session.im_list))))
        table.set_tag_filter(self._active_tags, self._include_untagged)
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
        table_index: int | None = None

        for idx in range(self.tabs.count()):
            if self.tabs.tabText(idx) == tab_title:
                widget = self.tabs.widget(idx)
                if isinstance(widget, _AudioTable):
                    table = widget
                    table_index = idx
                else:
                    table = _AudioTable(session, self)
                    old_widget = widget
                    self.tabs.removeTab(idx)
                    self._unregister_tab(old_widget)
                    table_index = self.tabs.insertTab(idx, table, tab_title)
                    self._configure_table(table)
                break

        if table is None:
            table = _AudioTable(session, self)
            table_index = self.tabs.addTab(table, tab_title)
            self._configure_table(table)

        if table_index is None:
            table_index = self.tabs.indexOf(table)
        self._register_tab(table, True, index=table_index)
        ref_idx = session.edge_to_song_index.get(ref_name)
        table.set_similarity(results, ref_index=ref_idx, ref_name=ref_name)
        table.set_tag_filter(self._active_tags, self._include_untagged)
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
        for widget in list(self._closable_tabs):
            widget.deleteLater()
        self._closable_tabs.clear()
        self.tabs.clear()
        self._sessions.clear()
        self._history.clear()
        self._hist_pos = -1
        self.similarityPairSelected.emit(None)

    # Methods used by existing code -----------------------------------
    def update_images(self, idxs: List[int], **kwargs) -> None:  # type: ignore[override]
        widget = self.tabs.currentWidget()
        if isinstance(widget, _AudioTable):
            widget.set_indices(idxs)
            self._history = self._history[: self._hist_pos + 1]
            self._history.append(idxs)
            self._hist_pos += 1
            self.historyChanged.emit()
        self.similarityPairSelected.emit(None)            

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
        self.similarityPairSelected.emit(None)

    def _configure_table(self, table: _AudioTable) -> None:
        table.selectionChanged.connect(partial(self._handle_table_selection, table))


    def _is_tab_index_closable(self, index: int) -> bool:
        widget = self.tabs.widget(index)
        return bool(widget and self._closable_tabs.get(widget, False))

    def _register_tab(self, widget: QWidget, closable: bool, *, index: int | None = None) -> None:
        self._closable_tabs[widget] = closable
        if index is None:
            index = self.tabs.indexOf(widget)
        if index != -1:
            self._update_close_button(index)

    def _unregister_tab(self, widget: QWidget | None) -> None:
        if widget is None:
            return
        self._closable_tabs.pop(widget, None)
        widget.deleteLater()

    def _update_close_button(self, index: int) -> None:
        if index < 0:
            return
        widget = self.tabs.widget(index)
        closable = bool(widget and self._closable_tabs.get(widget, False))
        tab_bar = self.tabs.tabBar()
        existing = tab_bar.tabButton(index, QTabBar.RightSide)
        if closable:
            if not isinstance(existing, QToolButton):
                button = QToolButton(self.tabs)
                button.setAutoRaise(True)
                button.setIcon(self.style().standardIcon(QStyle.SP_TitleBarCloseButton))
                button.setToolTip("Close tab")
                button.clicked.connect(partial(self._request_close_from_button, widget))
                tab_bar.setTabButton(index, QTabBar.RightSide, button)
        else:
            if isinstance(existing, QToolButton):
                existing.deleteLater()
            tab_bar.setTabButton(index, QTabBar.RightSide, None)

    def _request_close_from_button(self, widget: QWidget) -> None:
        index = self.tabs.indexOf(widget)
        if index != -1:
            self._close_tab(index)

    def _close_tab(self, index: int) -> None:
        if not self._is_tab_index_closable(index):
            return
        widget = self.tabs.widget(index)
        title = self.tabs.tabText(index)
        self.tabs.removeTab(index)
        self._sessions.pop(title, None)
        self._unregister_tab(widget)
        self.similarityPairSelected.emit(None)

    def _handle_table_selection(self, table: _AudioTable, indices: List[int]) -> None:
        unique = sorted({int(i) for i in indices})
        self.bus.set_images(unique)

        payload: Optional[dict] = None
        if table.is_similarity_tab() and len(unique) == 1:
            ref_idx = table.similarity_reference_index()
            match_idx = unique[0]
            if ref_idx is not None:
                match = table.best_segment_match(match_idx)
                if match:
                    session = table.session
                    ref_name = table.similarity_reference_name() or Path(
                        session.im_list[match.song_a]
                    ).name
                    match_name = Path(session.im_list[match.song_b]).name
                    payload = {
                        "reference_index": int(match.song_a),
                        "comparison_index": int(match.song_b),
                        "reference_start_s": float(match.start_a),
                        "comparison_start_s": float(match.start_b),
                        "reference_end_s": float(match.end_a),
                        "comparison_end_s": float(match.end_b),
                        "reference_name": ref_name,
                        "comparison_name": match_name,
                        "similarity": float(match.score),
                    }

        if payload is None:
            self.similarityPairSelected.emit(None)
        else:
            self.similarityPairSelected.emit(payload)



    def set_tag_filters(self, allowed: Iterable[str], include_untagged: bool) -> None:
        self._active_tags = {str(tag) for tag in allowed}
        self._include_untagged = bool(include_untagged)
        for table in self._iter_tables():
            table.set_tag_filter(self._active_tags, self._include_untagged)

    def refresh_tags(self, indices: Sequence[int]) -> None:
        idxs = [int(i) for i in indices]
        for table in self._iter_tables():
            table.refresh_tags_for_indices(idxs)

    def current_table(self) -> _AudioTable | None:
        widget = self.tabs.currentWidget()
        return widget if isinstance(widget, _AudioTable) else None

    def selected_song_indices(self) -> List[int]:
        table = self.current_table()
        return table.selected_song_indices() if table else []

    def _iter_tables(self):
        for idx in range(self.tabs.count()):
            widget = self.tabs.widget(idx)
            if isinstance(widget, _AudioTable):
                yield widget            