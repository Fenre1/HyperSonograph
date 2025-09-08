# prototype.py --------------------------------------------------------------
import sys, uuid, numpy as np
import pandas as pd
import numpy as np
from utils.similarity import SIM_METRIC
from pathlib import Path
import io
import torch
import time
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QApplication,
    QTreeView,
    QMainWindow,
    QFileDialog,
    QVBoxLayout,
    QWidget,
    QLabel,
    QSlider,
    QMessageBox,
    QPushButton,
    QDockWidget,
    QStackedWidget,
    QAction,
    QInputDialog,
    QDialog,
    QListWidget,
    QDialogButtonBox,
    QAbstractItemView,
    QLineEdit,
    QGroupBox,
    QCheckBox,
    QHBoxLayout,
)
 
from PyQt5.QtGui import (
    QStandardItem,
    QStandardItemModel,
    QPalette,
    QColor,
    QIcon,
    QPixmap,
)
from PyQt5.QtCore import (
    Qt,
    QSignalBlocker,
    QObject,
    pyqtSignal as Signal,
    QTimer,
    QSortFilterProxyModel,
    QRectF
)

from utils.data_loader import (
    DATA_DIRECTORY, get_h5_files_in_directory, load_session_data
)
from utils.selection_bus import SelectionBus
from utils.session_model import SessionModel, generate_n_colors
from utils.audio_table import AudioTableDock
from utils.audio_schema import SegmentLevel, SongLevel, ModelFeatures

from utils.spatial_viewQv4 import SpatialViewQDock, HyperedgeItem
from utils.feature_extraction import (
    AudioFeatureExtractorBase,    
    CLAPFeatureExtractor,
    MERTFeatureExtractor,
    OpenL3FeatureExtractor,
)
from utils.file_utils import get_audio_files
from utils.session_stats import show_session_stats
from utils.metadata_overview import show_metadata_overview
from utils.hyperedge_matrix2 import HyperedgeMatrixDock

from clustering.temi_clustering import temi_cluster

import pyqtgraph as pg
try:
    import darkdetect
    SYSTEM_DARK_MODE = darkdetect.isDark()
except Exception:
    SYSTEM_DARK_MODE = False

def apply_dark_palette(app: QApplication) -> None:
    """Apply a dark color palette to the given QApplication."""
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)

THRESHOLD_DEFAULT = 0.8
JACCARD_PRUNE_DEFAULT = 0.5
ORIGIN_COL = 3
SIM_COL = 4
STDDEV_COL = 5
INTER_COL = 6
DECIMALS = 3
UNGROUPED = "Ungrouped"

class _MultiSelectDialog(QDialog):
    """Simple dialog presenting a list for multi-selection."""

    def __init__(self, items: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Hyperedges")
        self.list = QListWidget()
        self.list.addItems(items)
        self.list.setSelectionMode(QAbstractItemView.MultiSelection)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.addWidget(self.list)
        lay.addWidget(buttons)

    def chosen(self) -> list[str]:
        return [it.text() for it in self.list.selectedItems()]


class HyperedgeSelectDialog(QDialog):
    """Dialog allowing the user to pick a hyperedge from a list with filtering."""

    def __init__(self, names: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Hyperedge")
        layout = QVBoxLayout(self)

        self.filter_edit = QLineEdit(self)
        self.filter_edit.setPlaceholderText("Filter...")
        layout.addWidget(self.filter_edit)

        self.list_widget = QListWidget(self)
        self.list_widget.addItems(names)
        self.list_widget.setSelectionMode(QListWidget.SingleSelection)
        layout.addWidget(self.list_widget)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.filter_edit.textChanged.connect(self._apply_filter)
        self.list_widget.itemDoubleClicked.connect(lambda *_: self.accept())

    def _apply_filter(self, text: str) -> None:
        text = text.lower()
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setHidden(text not in item.text().lower())

    def selected_name(self) -> str | None:
        items = self.list_widget.selectedItems()
        return items[0].text() if items else None


class NewSessionDialog(QDialog):
    """Dialog to set parameters for a new session."""

    def __init__(self, image_count: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Session")

        layout = QVBoxLayout(self)
        info = QLabel(
            f"Found {image_count} audio files.\n\n"
            "Feature extraction and clustering will be performed to generate "
            "the hypergraph. This may take a couple of minutes."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        layout.addWidget(QLabel("Number of hyperedges (20-200 recommended):"))
        self.edge_edit = QLineEdit("50")
        layout.addWidget(self.edge_edit)

        layout.addWidget(QLabel("Threshold for hypergraph generation:"))
        self.thr_edit = QLineEdit("0.5")
        layout.addWidget(self.thr_edit)
        thr_info = QLabel(
            "You can change this threshold later. Adjusting it is fast."
        )
        thr_info.setWordWrap(True)
        layout.addWidget(thr_info)

        layout.addWidget(QLabel("Duplicate removal Jaccard threshold:"))
        self.jacc_edit = QLineEdit(str(JACCARD_PRUNE_DEFAULT))
        layout.addWidget(self.jacc_edit)

        self.clap_cb = QCheckBox("Include CLAP model")
        self.mert_cb = QCheckBox("Include MERT model")
        self.openl3_cb = QCheckBox("Include OpenL3 model")
        layout.addWidget(self.clap_cb)
        layout.addWidget(self.mert_cb)
        layout.addWidget(self.openl3_cb)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Start generating hypergraph")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def parameters(self) -> tuple[int, float, float, list[str]]:
        models: list[str] = []
        if self.clap_cb.isChecked():
            models.append("CLAP")
        if self.mert_cb.isChecked():
            models.append("MERT")
        if self.openl3_cb.isChecked():
            models.append("OpenL3")
        return (
            int(self.edge_edit.text()),
            float(self.thr_edit.text()),
            float(self.jacc_edit.text()),
            models,
        )

class ReconstructDialog(QDialog):
    """Dialog to set parameters for hypergraph reconstruction."""

    def __init__(self, current_edges: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Reconstruct Hypergraph")

        layout = QVBoxLayout(self)
        info = QLabel(
            "Recalculate the clustering using the existing features."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        layout.addWidget(QLabel("Number of hyperedges:"))
        self.edge_edit = QLineEdit(str(current_edges))
        layout.addWidget(self.edge_edit)

        layout.addWidget(QLabel("Threshold for hypergraph generation:"))
        self.thr_edit = QLineEdit("0.5")
        layout.addWidget(self.thr_edit)

        layout.addWidget(QLabel("Duplicate removal Jaccard threshold:"))
        self.jacc_edit = QLineEdit(str(JACCARD_PRUNE_DEFAULT))
        layout.addWidget(self.jacc_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def parameters(self) -> tuple[int, float, float]:
        return (
            int(self.edge_edit.text()),
            float(self.thr_edit.text()),
            float(self.jacc_edit.text()),
        )



class HyperEdgeTree(QTreeView):
    """Navigator tree that lists meta-groups and individual hyper-edges."""
    def __init__(self, bus: SelectionBus, parent=None):
        super().__init__(parent)
        self.bus = bus
        # everything else (uniformRowHeights, sortingEnabled, etc.) stays
        self.setUniformRowHeights(True)
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(True)
        self.setSelectionBehavior(QTreeView.SelectRows)

    def _send_bus_update(self, *_):
        # column-0 indexes (name column)
        names = [idx.data(Qt.DisplayRole)
                 for idx in self.selectionModel().selectedRows(0)]
        self.bus.set_edges(names)


class TreeFilterProxyModel(QSortFilterProxyModel):
    """Proxy model to filter hyperedge tree items."""

    def filterAcceptsRow(self, source_row: int, source_parent) -> bool:  # type: ignore[override]
        if super().filterAcceptsRow(source_row, source_parent):
            return True

        model = self.sourceModel()
        index = model.index(source_row, 0, source_parent)
        for r in range(model.rowCount(index)):
            if self.filterAcceptsRow(r, index):
                return True
        return False


def _make_item(text: str = "", value=None, editable: bool = False):
    it = QStandardItem(text)
    it.setData(value, Qt.UserRole)

    if editable:
        it.setFlags(it.flags() |  Qt.ItemIsEditable)
    else:
        it.setFlags(it.flags() & ~Qt.ItemIsEditable)

    return it


def build_qmodel(rows, headers):
    model   = QStandardItemModel()
    model.setHorizontalHeaderLabels(headers)
    parents = {}

    for r in rows:
        g = r["group_name"]

        # ——— A. special-case “Ungrouped”: insert leaf at root
        if g == UNGROUPED:
            _append_leaf(model, r)
            continue

        # ——— B. ordinary meta-group
        if g not in parents:
            group_items = [_make_item(g)] + [_make_item() for _ in headers[1:]]
            parents[g] = group_items[0]
            model.appendRow(group_items)

        _append_leaf(parents[g], r)

    return model


def _append_leaf(parent_or_model, rowdict):
    """Add one leaf row under either a QStandardItem (group) or the model root."""
    container = parent_or_model
    name_item = _make_item(rowdict["name"], rowdict["name"], editable=True)
    name_item.setCheckable(True)
    name_item.setCheckState(Qt.Checked)    
    color = rowdict.get("color")
    if color:
        pix = QPixmap(12, 12)
        pix.fill(QColor(color))
        name_item.setIcon(QIcon(pix))
    leaf = [
        name_item,
        _make_item(str(rowdict["image_count"]), rowdict["image_count"]),
        _make_item(rowdict["status"]),
        _make_item(rowdict.get("origin", "")),
        _make_item(
            "" if rowdict["similarity"] is None
            else f"{rowdict['similarity']:.3f}",
            None if rowdict["similarity"] is None else float(rowdict["similarity"]),
        ),
        _make_item(
            "" if rowdict.get("stddev") is None else f"{rowdict['stddev']:.3f}",
            None if rowdict.get("stddev") is None else float(rowdict["stddev"]),
        ),
        _make_item(
            "" if rowdict.get("intersection") is None else str(rowdict["intersection"]),
            rowdict.get("intersection"),
        ),
    ]
    container.appendRow(leaf)

def calculate_similarity_matrix(vecs):
    names = list(vecs)
    if not names:
        return pd.DataFrame()
    m = np.array(list(vecs.values()))
    s = SIM_METRIC(m, m)
    np.fill_diagonal(s, -np.inf)
    return pd.DataFrame(s, index=names, columns=names)


def perform_hierarchical_grouping(model, thresh=0.8):
    vecs   = model.hyperedge_avg_features.copy()
    comp   = {k: [k] for k in vecs}
    counts = {k: 1     for k in vecs}

    while len(vecs) > 1:
        sim = calculate_similarity_matrix(vecs)
        if sim.empty or sim.values.max() < thresh:
            break
        col = sim.max().idxmax()
        row = sim[col].idxmax()

        new = f"temp_{uuid.uuid4()}"
        c1, c2 = counts[row], counts[col]
        vecs[new] = (vecs[row] * c1 + vecs[col] * c2) / (c1 + c2)
        comp[new] = comp.pop(row) + comp.pop(col)
        counts[new] = c1 + c2
        vecs.pop(row), vecs.pop(col)
    return comp


def rename_groups_sequentially(raw):
    res, cnt, singles = {}, 1, []
    for k, ch in raw.items():
        if len(ch) > 1:
            res[f"Meta-Group {cnt}"] = ch
            cnt += 1
        else:
            singles.extend(ch)
    if singles:
        res["Ungrouped"] = singles
    return res


def build_row_data(groups, model):
    status = model.status_map
    rows = []
    for g, children in groups.items():
        for child in children:
            meta = status[child]
            stddev = model.similarity_std(child)
            rows.append(
                dict(
                    uuid=meta["uuid"],
                    name=child,
                    image_count=len(model.hyperedges[child]),
                    status=meta["status"],
                    origin=model.edge_origins.get(child, ""),
                    similarity=None,
                    stddev=stddev,
                    intersection=None,
                    group_name=g,
                    color=model.edge_colors.get(child, "#808080"),
                )
            )
    return rows



class MainWin(QMainWindow):
    def _source_model(self):
        model = self.tree.model()
        return model.sourceModel() if isinstance(model, QSortFilterProxyModel) else model

    def _source_index(self, index):
        model = self.tree.model()
        return model.mapToSource(index) if isinstance(model, QSortFilterProxyModel) else index

    def _item_from_index(self, index):
        src_model = self._source_model()
        src_index = self._source_index(index)
        return src_model.itemFromIndex(src_index)


    def _vector_for(self, name: str) -> np.ndarray | None:
        avg = self.model.hyperedge_avg_features
        if name in avg:
            return avg[name][None, :]

        if name in self.groups:
            child_vecs = [avg[c] for c in self.groups[name] if c in avg]
            if child_vecs:
                return np.mean(child_vecs, axis=0, keepdims=True)
        return None

    def compute_similarity(self, ref_name: str | None = None):
        if not self.model:
            return
        if ref_name is None:
            sel = self.tree.selectionModel().selectedRows(0)
            if not sel:
                return
            ref_name = sel[0].data(Qt.DisplayRole)

        if ref_name == False:
            sel = self.tree.selectionModel().selectedRows(0)
            if not sel:
                return
            ref_name = sel[0].data(Qt.DisplayRole)            
        ref_vec = self._vector_for(ref_name)
        if ref_vec is None:
            QMessageBox.warning(self, "No features", f"No feature vector for “{ref_name}”.")
            return

        avg = self.model.hyperedge_avg_features
        names, vectors = list(avg), np.stack(list(avg.values()))
        sims = SIM_METRIC(ref_vec, vectors)[0]
        sim_map = dict(zip(names, sims))

        ref_imgs = self.model.hyperedges.get(ref_name, set())
        inter_map = {
            name: len(ref_imgs & self.model.hyperedges.get(name, set()))
            for name in names
        }

        model = self.tree.model()
        root = (
            model.sourceModel().invisibleRootItem()
            if isinstance(model, QSortFilterProxyModel)
            else model.invisibleRootItem()
        )
        self._update_similarity_items(root, sim_map)
        self._update_intersection_items(root, inter_map)
        self._source_model().sort(SIM_COL, Qt.DescendingOrder)
        self._similarity_ref = ref_name
        self._similarity_computed = True

    def _update_similarity_items(self, parent: QStandardItem, sim_map):
        for r in range(parent.rowCount()):
            name_item, sim_item = parent.child(r, 0), parent.child(r, SIM_COL)

            if name_item.hasChildren():
                self._update_similarity_items(name_item, sim_map)
                child_vals = [name_item.child(c, SIM_COL).data(Qt.UserRole) for c in range(name_item.rowCount())]
                val = np.nanmean([v for v in child_vals if v is not None]) if child_vals else None
            else:
                val = sim_map.get(name_item.text())

            if val is None:
                sim_item.setData(None, Qt.UserRole); sim_item.setData("", Qt.DisplayRole)
            else:
                sim_item.setData(float(val), Qt.UserRole); sim_item.setData(f"{val:.{DECIMALS}f}", Qt.DisplayRole)

    def _compute_overview_triplets(self) -> dict[str, tuple[int | None, ...]]:
        """Delegate to the session model for cached overview triplets."""
        if not self.model:
            return {}
        return self.model.compute_overview_triplets()


    def _update_intersection_items(self, parent: QStandardItem, inter_map):
        for r in range(parent.rowCount()):
            name_item = parent.child(r, 0)
            inter_item = parent.child(r, INTER_COL)

            if name_item.hasChildren():
                self._update_intersection_items(name_item, inter_map)
                child_vals = [name_item.child(c, INTER_COL).data(Qt.UserRole) for c in range(name_item.rowCount())]
                val = sum(v for v in child_vals if v is not None) if child_vals else None
            else:
                val = inter_map.get(name_item.text())

            if val is None:
                inter_item.setData(None, Qt.UserRole)
                inter_item.setData("", Qt.DisplayRole)
            else:
                inter_item.setData(int(val), Qt.UserRole)
                inter_item.setData(str(int(val)), Qt.DisplayRole)

    def __init__(self):
        super().__init__()
        self.setDockNestingEnabled(True)
        self.setWindowTitle("Hypergraph Desktop Prototype")
        self.resize(1200, 800)

        self.model = None
        self._clip_extractor = None
        self._openclip_extractor = None
        self._overview_triplets = None
        self.temi_results = {}
        self.bus = SelectionBus()
        # self.bus.edgesChanged.connect(self._update_bus_images)

        self.bus.edgesChanged.connect(self._remember_last_edge)

        self._last_edge = None
        self._similarity_ref = None
        self._similarity_computed = False
        # Track layout changes vs. hyperedge modifications
        self._skip_next_layout = False
        self._layout_timer = QTimer(self)
        self._layout_timer.setSingleShot(True)
        self._layout_timer.timeout.connect(self._apply_layout_change)
        self._skip_reset_timer = QTimer(self)
        self._skip_reset_timer.setSingleShot(True)
        self._skip_reset_timer.timeout.connect(lambda: setattr(self, "_skip_next_layout", False))

        # ----------------- WIDGET AND DOCK CREATION ------------------------------------
        # Create all widgets and docks first, then arrange them.

        # --- List Tree ---
        self.tree = HyperEdgeTree(self.bus)
        self.tree_proxy = TreeFilterProxyModel(self)
        self.tree_proxy.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.tree_proxy.setFilterKeyColumn(0)

        self.tree_filter = QLineEdit()
        self.tree_filter.setPlaceholderText("Filter hyperedges…")
        self.tree_filter.textChanged.connect(self.tree_proxy.setFilterFixedString)

        tree_container = QWidget()
        tree_layout = QVBoxLayout(tree_container)
        tree_layout.setContentsMargins(0, 0, 0, 0)
        tree_layout.addWidget(self.tree_filter)
        tree_layout.addWidget(self.tree)

        self.tree_dock = QDockWidget("List tree", self)
        self.tree_dock.setWidget(tree_container)

        # --- Buttons / Tools Dock ---
        self.toolbar_dock = QDockWidget("Buttons", self)
        toolbar_container = QWidget()
        toolbar_layout = QVBoxLayout(toolbar_container)
        toolbar_layout.setContentsMargins(10, 10, 10, 10)
        toolbar_layout.setSpacing(10)

        self.btn_sim = QPushButton("Show similarity and intersection")
        self.btn_sim.clicked.connect(self.compute_similarity)
        
        self.btn_add_hyperedge = QPushButton("Add Hyperedge")
        self.btn_add_hyperedge.clicked.connect(self.on_add_hyperedge)
        
        self.btn_del_hyperedge = QPushButton("Delete Hyperedge")
        self.btn_del_hyperedge.clicked.connect(self.on_delete_hyperedge)


        self.btn_overview = QPushButton("Overview")
        self.btn_overview.clicked.connect(self.show_overview)

        self.btn_meta_overview = QPushButton("Metadata overview")
        self.btn_meta_overview.clicked.connect(self.show_metadata_overview)

        self.btn_color_default = QPushButton("Colorize (edge)")
        self.btn_color_default.clicked.connect(self.color_edges_default)

        self.btn_color_status = QPushButton("Colorize by status")
        self.btn_color_status.clicked.connect(self.color_edges_by_status)

        self.btn_color_origin = QPushButton("Colorize by origin")
        self.btn_color_origin.clicked.connect(self.color_edges_by_origin)

        self.btn_color_similarity = QPushButton("Colorize by similarity")
        self.btn_color_similarity.clicked.connect(self.color_edges_by_similarity)

        self.btn_session_stats = QPushButton("Session stats")
        self.btn_session_stats.clicked.connect(self.show_session_stats)

        self.btn_manage_visibility = QPushButton("Manage hidden hyperedges")
        self.btn_manage_visibility.clicked.connect(self.choose_hidden_edges)

        # --- Spatial view limits ---
        self.limit_images_cb = QCheckBox("Limit number of image nodes per hyperedge")
        self.limit_images_edit = QLineEdit("10")
        lim_img_row = QHBoxLayout(); lim_img_row.addWidget(self.limit_images_cb); lim_img_row.addWidget(self.limit_images_edit)
        lim_img_w = QWidget(); lim_img_w.setLayout(lim_img_row)

        self.limit_edges_cb = QCheckBox("Limit number of intersecting hyperedges")
        self.limit_edges_edit = QLineEdit("10")
        lim_edge_row = QHBoxLayout(); lim_edge_row.addWidget(self.limit_edges_cb); lim_edge_row.addWidget(self.limit_edges_edit)
        lim_edge_w = QWidget(); lim_edge_row.setContentsMargins(0,0,0,0); lim_img_row.setContentsMargins(0,0,0,0); lim_edge_w.setLayout(lim_edge_row)



        toolbar_layout.addWidget(self.btn_sim)
        toolbar_layout.addWidget(self.btn_add_hyperedge)
        toolbar_layout.addWidget(self.btn_del_hyperedge)

        toolbar_layout.addWidget(self.btn_overview)
        toolbar_layout.addWidget(self.btn_session_stats)
        toolbar_layout.addWidget(self.btn_manage_visibility)        

        toolbar_layout.addWidget(self.btn_meta_overview)
        toolbar_layout.addWidget(self.btn_color_default)
        toolbar_layout.addWidget(self.btn_color_status)
        toolbar_layout.addWidget(self.btn_color_origin)
        toolbar_layout.addWidget(self.btn_color_similarity)
        toolbar_layout.addWidget(lim_img_w)
        toolbar_layout.addWidget(lim_edge_w)


        self.legend_box = QGroupBox("Legend")
        self.legend_layout = QVBoxLayout(self.legend_box)
        self.legend_layout.setContentsMargins(4, 4, 4, 4)
        self.legend_box.hide()
        toolbar_layout.addWidget(self.legend_box)

        self.audio_table = AudioTableDock(self.bus, self)
        toolbar_layout.addWidget(self.audio_table.hide_selected_cb)
        toolbar_layout.addWidget(self.audio_table.hide_modified_cb)

        toolbar_layout.addStretch()
        self.toolbar_dock.setWidget(toolbar_container)

        self.audio_table.setObjectName("AudioTableDock")

        # --- Spatial View ---
        self.spatial_dock = SpatialViewQDock(self.bus, self)
        self.spatial_dock.setObjectName("SpatialViewDock")
        self.limit_images_cb.toggled.connect(self._update_spatial_limits)
        self.limit_images_edit.editingFinished.connect(self._update_spatial_limits)
        self.limit_edges_cb.toggled.connect(self._update_spatial_limits)
        self.limit_edges_edit.editingFinished.connect(self._update_spatial_limits)
        self._update_spatial_limits()



        # --- Hyperedge Matrix ---
        self.matrix_dock = HyperedgeMatrixDock(self.bus, self)
        self.matrix_dock.setObjectName("HyperedgeMatrixDock")

        # --- Grouping Slider (no longer in central widget) ---
        # We can place these controls in one of the docks, e.g., the 'Buttons' dock.
        self.slider = QSlider(Qt.Horizontal, minimum=50, maximum=100, singleStep=5, value=int(THRESHOLD_DEFAULT * 100))
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        toolbar_layout.insertWidget(0, self.label)
        toolbar_layout.insertWidget(1, self.slider)


        # ----------------- DOCK LAYOUT ARRANGEMENT ------------------------------------
        # Arrange the created docks to match the wireframe.
        # No central widget is set, allowing docks to fill the entire window.

        # 1. Add the "List tree" to the left area.
        self.addDockWidget(Qt.LeftDockWidgetArea, self.tree_dock)

        # 2. Add the "Buttons" dock under the "List tree" dock.
        self.addDockWidget(Qt.LeftDockWidgetArea, self.toolbar_dock)

        # 3. Add the "Image grid" to the right area. It will take up the remaining space.
        self.addDockWidget(Qt.RightDockWidgetArea, self.audio_table)

        # 4. Add the "Spatial view" below the "Image grid".
        self.addDockWidget(Qt.RightDockWidgetArea, self.spatial_dock)

        # 5. Split the area occupied by the "Spatial view" to place the "Hyperedge matrix" to its right.
        self.splitDockWidget(self.spatial_dock, self.matrix_dock, Qt.Horizontal)

        # Optional: Set initial relative sizes of the docks
        self.resizeDocks([self.tree_dock, self.audio_table], [300, 850], Qt.Horizontal)
        self.resizeDocks([self.tree_dock, self.toolbar_dock], [550, 250], Qt.Vertical)
        self.resizeDocks([self.spatial_dock, self.matrix_dock], [450, 450], Qt.Horizontal)


        # ----------------- MENU AND STATE ------------------------------------
        open_act = QAction("&Open Session…", self, triggered=self.open_session)
        new_act = QAction("&New Session…", self, triggered=self.new_session)
        save_act = QAction("&Save", self, triggered=self.save_session)
        save_as_act = QAction("Save &As…", self, triggered=self.save_session_as)

        reconstruct_act = QAction("Reconstruct Hypergraph…", self,
                                   triggered=self.reconstruct_hypergraph)

        self.thumb_toggle_act = QAction("Use Full Images", self, checkable=True)
        self.thumb_toggle_act.toggled.connect(self.toggle_full_images)

        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(open_act)
        file_menu.addAction(new_act)
        file_menu.addAction(save_act)
        file_menu.addAction(save_as_act)
        file_menu.addAction(self.thumb_toggle_act)
        file_menu.addAction(reconstruct_act)       
        file_menu.addAction(reconstruct_act)
        import_folder_act = QAction("Import folder", self, triggered=self.import_folder)
        file_menu.addAction(import_folder_act)         
        # self.menuBar().addMenu("&File").addAction(open_act)

        self.model = None

        self.temi_results = {}
        self.audio_files: list[str] = []
        self.features: dict[str, np.ndarray] = {}
        self._model_names: list[str] = []
        self._model_features: dict[str, ModelFeatures] = {}
        self.slider.valueChanged.connect(self.regroup)



    def import_folder(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select folder")
        if not directory:
            return
        path = Path(directory)
        files = [str(p) for p in path.rglob("*") if p.suffix.lower() in {".wav", ".mp3"}]
        if not files:
            QMessageBox.information(self, "No Audio", "No .wav or .mp3 files found.")
            return
        msg = QMessageBox(self)
        msg.setWindowTitle("Process Audio")
        msg.setText(f"Found {len(files)} audio files. Process?")
        process_btn = msg.addButton("Process", QMessageBox.AcceptRole)
        msg.addButton(QMessageBox.Cancel)
        msg.exec_()
        if msg.clickedButton() is process_btn:
            self.process_audio_files(files)

    def process_audio_files(self, files: list[str]) -> None:
        """Import additional audio files into the current session.

        This performs feature extraction at both the segment and song level for
        all available models. The resulting :class:`ModelFeatures` objects are
        appended to ``_model_features`` and the hypergraph is reconstructed so
        that the UI reflects the newly added songs.
        """

        if not self.model:
            QMessageBox.warning(
                self, "No Session", "Please load or create a session first."
            )
            return

        extractors: list[AudioFeatureExtractorBase] = []
        try:
            extractors.append(CLAPFeatureExtractor())
        except Exception:
            pass
        try:
            extractors.append(MERTFeatureExtractor())
        except Exception:
            pass
        try:
            extractors.append(OpenL3FeatureExtractor())
        except Exception:
            pass
        if not extractors:
            QMessageBox.warning(
                self, "No Models", "No audio feature extractors available."
            )
            return

        start_idx = len(self.model.im_list)
        file_index = {f: start_idx + i for i, f in enumerate(files)}

        for ext in extractors:
            Xseg, rows, songs = ext.extract_segments_and_songs(files)

            seg_song_ids = np.array(
                [file_index[r["file"]] for r in rows], dtype=np.int32
            )
            segs = SegmentLevel(
                embeddings=Xseg,
                song_id=seg_song_ids,
                start_s=np.array([r["start_s"] for r in rows], dtype=np.float32),
                end_s=np.array([r["end_s"] for r in rows], dtype=np.float32),
            )
            centroids = (
                np.vstack([s.centroid_D for s in songs])
                if songs
                else np.zeros((0, ext.output_dim()))
            )
            stats = (
                np.vstack([s.stats_2D for s in songs])
                if songs
                else np.zeros((0, 2 * ext.output_dim()))
            )
            song_ids = np.array(
                [file_index[s.file] for s in songs], dtype=np.int32
            )
            paths = [s.file for s in songs]
            song_level = SongLevel(
                centroid=centroids,
                stats2D=stats,
                song_id=song_ids,
                path=paths,
            )
            mf = ModelFeatures(name=ext.model_name, segments=segs, songs=song_level)

            if ext.model_name in self._model_features:
                old = self._model_features[ext.model_name]
                merged_segments = SegmentLevel(
                    embeddings=np.vstack([old.segments.embeddings, segs.embeddings]),
                    song_id=np.concatenate([old.segments.song_id, segs.song_id]),
                    start_s=np.concatenate([old.segments.start_s, segs.start_s]),
                    end_s=np.concatenate([old.segments.end_s, segs.end_s]),
                )
                merged_songs = SongLevel(
                    centroid=np.vstack([old.songs.centroid, song_level.centroid]),
                    stats2D=np.vstack([old.songs.stats2D, song_level.stats2D]),
                    song_id=np.concatenate([old.songs.song_id, song_level.song_id]),
                    path=list(old.songs.path) + list(song_level.path),
                )
                self._model_features[ext.model_name] = ModelFeatures(
                    name=ext.model_name, segments=merged_segments, songs=merged_songs
                )
            else:
                self._model_features[ext.model_name] = mf
                self._model_names.append(ext.model_name)

        features = self._model_features[self._model_names[0]].songs.stats2D
        oc_feats = (
            self._model_features[self._model_names[1]].songs.stats2D
            if len(self._model_names) > 1
            else None
        )
        plc_feats = (
            self._model_features[self._model_names[2]].songs.stats2D
            if len(self._model_names) > 2
            else None
        )

        n_edges = len(self.model.cat_list)
        app = QApplication.instance()
        if app:
            app.setOverrideCursor(Qt.WaitCursor)
        try:
            matrix, _ = temi_cluster(
                features, out_dim=n_edges, threshold=THRESHOLD_DEFAULT
            )
            oc_matrix = None
            if oc_feats is not None:
                oc_matrix, _ = temi_cluster(
                    oc_feats, out_dim=n_edges, threshold=THRESHOLD_DEFAULT
                )
            plc_matrix = None
            if plc_feats is not None:
                plc_matrix, _ = temi_cluster(
                    plc_feats, out_dim=n_edges, threshold=THRESHOLD_DEFAULT
                )

            empty_cols = np.where(matrix.sum(axis=0) == 0)[0]
            if len(empty_cols) > 0:
                matrix = np.delete(matrix, empty_cols, axis=1)
            if oc_matrix is not None:
                oc_empty = np.where(oc_matrix.sum(axis=0) == 0)[0]
                if len(oc_empty) > 0:
                    oc_matrix = np.delete(oc_matrix, oc_empty, axis=1)
            if plc_matrix is not None:
                plc_empty = np.where(plc_matrix.sum(axis=0) == 0)[0]
                if len(plc_empty) > 0:
                    plc_matrix = np.delete(plc_matrix, plc_empty, axis=1)
        except Exception as e:
            if app:
                app.restoreOverrideCursor()
            QMessageBox.critical(self, "Processing Error", str(e))
            return
        if app:
            app.restoreOverrideCursor()

        df = pd.DataFrame(
            matrix.astype(int),
            columns=[f"edge_{i}" for i in range(matrix.shape[1])],
        )

        new_files = self.model.im_list + files
        metadata = self.model.metadata
        if metadata is not None:
            metadata = pd.concat(
                [metadata, pd.DataFrame(index=np.arange(len(files)))],
                ignore_index=True,
            )

        try:
            self.model.layoutChanged.disconnect(self.regroup)
        except Exception:
            pass
        try:
            self.model.layoutChanged.disconnect(self._on_layout_changed)
        except Exception:
            pass
        try:
            self.model.hyperedgeModified.disconnect(
                self._on_model_hyperedge_modified
            )
        except Exception:
            pass

        self.model = SessionModel(
            new_files,
            df,
            features,
            self.model.h5_path,
            openclip_features=oc_feats,
            places365_features=plc_feats,
            thumbnail_data=self.model.thumbnail_data,
            thumbnails_are_embedded=self.model.thumbnails_are_embedded,
            metadata=metadata,
        )

        if oc_feats is not None and oc_feats.size and oc_matrix is not None:
            origin = self._model_names[1].lower()
            self.model.append_clustering_matrix(oc_matrix, origin=origin, prefix=origin)
        if plc_feats is not None and plc_feats.size and plc_matrix is not None:
            origin = self._model_names[2].lower()
            self.model.append_clustering_matrix(plc_matrix, origin=origin, prefix=origin)
        self.model.prune_similar_edges(JACCARD_PRUNE_DEFAULT)

        self.model.layoutChanged.connect(self.regroup)
        self._overview_triplets = None
        self.model.layoutChanged.connect(self._on_layout_changed)
        self.model.hyperedgeModified.connect(
            self._on_model_hyperedge_modified
        )

        self.audio_table.clear()
        for name in self._model_names:
            self.audio_table.add_model(name, self.model)
        self.audio_table.set_use_full_images(True)
        self.thumb_toggle_act.setChecked(True)
        self.matrix_dock.set_model(self.model)
        self.spatial_dock.set_model(self.model)
        self.regroup()

        self.audio_files.extend(files)



    def on_add_hyperedge(self):
        if not self.model:
            QMessageBox.warning(self, "No Session", "Please load a session first.")
            return

        # Use QInputDialog to get text from the user
        new_name, ok = QInputDialog.getText(self, "Add New Hyperedge", "Enter name for the new hyperedge:")

        if ok and new_name:
            # User clicked OK and entered text
            clean_name = new_name.strip()

            # --- Validation ---
            if not clean_name:
                QMessageBox.warning(self, "Invalid Name", "Hyperedge name cannot be empty.")
                return

            if clean_name in self.model.hyperedges:
                QMessageBox.warning(self, "Duplicate Name",
                                    f"A hyperedge named '{clean_name}' already exists.")
                return

            # --- Call the model method to perform the addition ---
            self.model.add_empty_hyperedge(clean_name)
            # The model will emit layoutChanged, which is connected to self.regroup in load_session
        else:
            # User clicked Cancel or entered nothing
            print("Add hyperedge cancelled.")

    def on_delete_hyperedge(self):
        if not self.model:
            QMessageBox.warning(self, "No Session", "Please load a session first.")
            return

        model = self.tree.model()
        sel = self.tree.selectionModel().selectedRows(0) if model else []
        if not sel:
            QMessageBox.information(self, "No Selection", "Select a hyperedge in the tree first.")
            return

        item = self._item_from_index(sel[0])
        if item.hasChildren():
            QMessageBox.information(self, "Invalid Selection", "Please select a single hyperedge, not a group.")
            return

        name = item.text()
        res = QMessageBox.question(self, "Delete Hyperedge", f"Delete hyperedge '{name}'?", QMessageBox.Yes | QMessageBox.No)
        if res == QMessageBox.Yes:
            self.model.delete_hyperedge(name)


    def show_overview(self):
        """Display triplet overview on the image grid."""
        if not self.model:
            QMessageBox.warning(self, "No Session", "Please load a session first.")
            return
        if self._overview_triplets is None:
            self._overview_triplets = self._compute_overview_triplets()
        self.audio_table.show_overview(self._overview_triplets, self.model)


    def show_metadata_overview(self):
        """Show a summary of all metadata in a popup window."""
        if not self.model:
            QMessageBox.warning(self, "No Session", "Please load a session first.")
            return
        show_metadata_overview(self.model, self)


    def show_session_stats(self):
        """Show basic statistics about the current session."""
        if not self.model:
            QMessageBox.warning(self, "No Session", "Please load a session first.")
            return
        show_session_stats(self.model, self)

    def add_metadata_hyperedges(self, column: str) -> None:
        """Create hyperedges from a metadata column."""
        if not self.model:
            QMessageBox.warning(self, "No Session", "Please load a session first.")
            return
        if column not in self.model.metadata.columns:
            QMessageBox.warning(self, "Unknown Metadata", f"No metadata column '{column}'.")
            return

        series = self.model.metadata[column]
        strs = series.astype(str)
        valid_mask = series.notna() & strs.str.strip().ne("") & ~strs.str.lower().isin(["none", "nan"])
        if column in self.model.hyperedges:
            QMessageBox.warning(self, "Duplicate Hyperedge", f"Hyperedge '{column}' already exists.")
            return

        self._skip_next_layout = True
        self.model.add_empty_hyperedge(column)
        self.model.edge_origins[column] = "Metadata"
        # self.model.edge_colors[column] = "#000000"
        self.model.add_images_to_hyperedge(column, series[valid_mask].index.tolist())

        categorical = True
        valid_values = strs[valid_mask].tolist()
        if valid_values:
            try:
                [float(v) for v in valid_values]
                categorical = False
            except Exception:
                categorical = True
        sub_edges = []
        if categorical:
            unique_vals = sorted({str(v) for v in strs[valid_mask]})
            for val in unique_vals:
                name = f"{val} {column}"
                self.model.add_empty_hyperedge(name)
                self.model.edge_origins[name] = "Metadata"
                # self.model.edge_colors[name] = "#808080"
                mask = valid_mask & (strs == val)
                self.model.add_images_to_hyperedge(name, series[mask].index.tolist())
                sub_edges.append(name)

        if hasattr(self, "spatial_dock") and self.spatial_dock.fa2_layout:
            fa = self.spatial_dock.fa2_layout
            pos = np.array(list(fa.positions.values()))
            max_x = pos[:, 0].max() if pos.size else 0.0
            max_y = pos[:, 1].max() if pos.size else 0.0
            new_pos = np.array([max_x * 1.1, max_y])
            all_edges = [column] + sub_edges
            for name in all_edges:
                size = max(np.sqrt(len(self.model.hyperedges[name])) * self.spatial_dock.NODE_SIZE_SCALER,
                           self.spatial_dock.MIN_HYPEREDGE_DIAMETER)
                fa.node_sizes = np.append(fa.node_sizes, size)
                fa.names.append(name)
                fa.positions[name] = new_pos.copy()
                self.spatial_dock.edge_index[name] = len(fa.names) - 1
                ell = HyperedgeItem(name, QRectF(-size/2, -size/2, size, size))
                # col = "#000000" if name == column else "#808080"
                col = self.model.edge_colors.get(name, "#000000")
                ell.setPen(pg.mkPen(col))
                ell.setBrush(pg.mkBrush(col))
                self.spatial_dock.view.addItem(ell)
                ell.setPos(*new_pos)
                self.spatial_dock.hyperedgeItems[name] = ell
            self.spatial_dock._refresh_edges()
            self.spatial_dock._update_image_layer()
        self._skip_next_layout = False

    # ------------------------------------------------------------------



    def new_session(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select audio folder", str(DATA_DIRECTORY)
        )
        if not directory:
            return

        files = get_audio_files(directory)
        if not files:
            QMessageBox.information(
                self, "No Audio", "No supported audio files found."
            )
            return

        dlg = NewSessionDialog(len(files), self)
        if dlg.exec_() != QDialog.Accepted:
            return
        try:
            n_edges, thr, prune_thr, models = dlg.parameters()
        except Exception:
            QMessageBox.warning(self, "Invalid Input", "Enter valid numbers.")
            return
        if not models:
            QMessageBox.warning(self, "No Model", "Select at least one model.")
            return


        app = QApplication.instance()
        if app:
            app.setOverrideCursor(Qt.WaitCursor)
        try:
            matrices: dict[str, np.ndarray] = {}
            features_by_model: dict[str, np.ndarray] = {}
            model_feats: dict[str, ModelFeatures] = {}
            file_index = {f: i for i, f in enumerate(files)}
            for name in models:
                if name == "CLAP":
                    ext = CLAPFeatureExtractor()
                elif name == "MERT":
                    ext = MERTFeatureExtractor()
                else:
                    ext = OpenL3FeatureExtractor()
                Xseg, rows, songs = ext.extract_segments_and_songs(files)
                seg_song_ids = np.array([file_index[r["file"]] for r in rows], dtype=np.int32)
                segs = SegmentLevel(
                    embeddings=Xseg,
                    song_id=seg_song_ids,
                    start_s=np.array([r["start_s"] for r in rows], dtype=np.float32),
                    end_s=np.array([r["end_s"] for r in rows], dtype=np.float32),
                )
                centroids = np.vstack([s.centroid_D for s in songs]) if songs else np.zeros((0, ext.output_dim()))
                stats = np.vstack([s.stats_2D for s in songs]) if songs else np.zeros((0, 2 * ext.output_dim()))
                song_level = SongLevel(
                    centroid=centroids,
                    stats2D=stats,
                    song_id=np.arange(len(files)),
                    path=files,
                )
                mf = ModelFeatures(name=name, segments=segs, songs=song_level)
                model_feats[name] = mf
                feats = song_level.stats2D
                features_by_model[name] = feats
                matrix, _ = temi_cluster(feats, out_dim=n_edges, threshold=thr)
                empty = np.where(matrix.sum(axis=0) == 0)[0]
                if len(empty) > 0:
                    matrix = np.delete(matrix, empty, axis=1)
                    if name == models[0]:
                        QMessageBox.information(
                            self,
                            "Empty Hyperedges Removed",
                            f"{len(empty)} empty hyperedges were removed after clustering.",
                        )
                matrices[name] = matrix
        except Exception as e:
            if app:
                app.restoreOverrideCursor()
            QMessageBox.critical(self, "Generation Error", str(e))
            return
        if app:
            app.restoreOverrideCursor()

        matrix = matrices[models[0]]
        features = features_by_model[models[0]]
        oc_matrix = matrices.get(models[1]) if len(models) > 1 else np.array([])
        oc_features = features_by_model.get(models[1]) if len(models) > 1 else None
        plc_matrix = matrices.get(models[2]) if len(models) > 2 else np.array([])
        plc_features = features_by_model.get(models[2]) if len(models) > 2 else None
        self._model_features = model_feats
        self._model_names = models
        df = pd.DataFrame(matrix.astype(int), columns=[f"edge_{i}" for i in range(matrix.shape[1])])

        if self.model:
            try:
                self.model.layoutChanged.disconnect(self.regroup)
            except TypeError:
                pass
            try:
                self.model.layoutChanged.disconnect(self._on_layout_changed)
            except TypeError:
                pass
            try:
                self.model.hyperedgeModified.disconnect(self._on_model_hyperedge_modified)
            except TypeError:
                pass

        self.model = SessionModel(
            files,
            df,
            features,
            Path(directory),
            openclip_features=oc_features,
            places365_features=plc_features,
            model_features=model_feats,
            model_names=models,
        )
        if oc_matrix.size:
            origin = models[1].lower()
            self.model.append_clustering_matrix(oc_matrix, origin=origin, prefix=origin)
        if plc_matrix.size:
            origin = models[2].lower()
            self.model.append_clustering_matrix(plc_matrix, origin=origin, prefix=origin)
        self.model.prune_similar_edges(prune_thr)


        self.model.layoutChanged.connect(self.regroup)
        self._overview_triplets = None
        self.model.layoutChanged.connect(self._on_layout_changed)
        self.model.hyperedgeModified.connect(self._on_model_hyperedge_modified)

        self.audio_table.clear()
        for name in self._model_names:
            self.audio_table.add_model(name, self.model)
        self.audio_table.set_use_full_images(True)
        self.thumb_toggle_act.setChecked(True)
        self.matrix_dock.set_model(self.model)
        self.spatial_dock.set_model(self.model)
        self.regroup()


    def open_session(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select .h5 session", DATA_DIRECTORY, "H5 files (*.h5)")
        if file: 
            self.load_session(Path(file))

    def load_session(self, path: Path):
        try:
            # If a model already exists, disconnect its signal first
            if self.model:
                try:
                    self.model.layoutChanged.disconnect(self.regroup)
                except TypeError:
                    pass
                try:
                    self.model.layoutChanged.disconnect(self._on_layout_changed)
                except TypeError:
                    pass
                try:
                    self.model.hyperedgeModified.disconnect(self._on_model_hyperedge_modified)
                except TypeError:
                    pass

            self.model = SessionModel.load_h5(path)
            self._model_features = getattr(self.model, "model_features", {})
            self._model_names = getattr(self.model, "model_names", list(self._model_features.keys()))

            self.model.layoutChanged.connect(self.regroup)
            self._overview_triplets = None
            self.model.layoutChanged.connect(self._on_layout_changed)
            self.model.hyperedgeModified.connect(self._on_model_hyperedge_modified)
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))
            return

        self.audio_table.clear()
        for name in self._model_names:
            self.audio_table.add_model(name, self.model)
        self.audio_table.set_use_full_images(True)
        self.thumb_toggle_act.setChecked(True)

        self.matrix_dock.set_model(self.model)
        self.spatial_dock.set_model(self.model)
        self.regroup()


    def save_session(self):
        if not self.model:
            QMessageBox.warning(self, "No Session", "Please load or create a session first.")
            return
        path = self.model.h5_path
        if not path or path.suffix.lower() != ".h5":
            return self.save_session_as()

        try:
            self.model.save_h5()
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def save_session_as(self):
        if not self.model:
            QMessageBox.warning(self, "No Session", "Please load or create a session first.")
            return
        base = self.model.h5_path
        if base and base.suffix:
            default_dir = str(base)
        else:
            default_dir = str(base if base else DATA_DIRECTORY)
        file, _ = QFileDialog.getSaveFileName(self, "Save Session As", default_dir, "H5 files (*.h5)")
        if not file:
            return
        try:
            self.model.save_h5(Path(file))
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def reconstruct_hypergraph(self):
        """Re-run clustering using existing features."""
        if not self.model:
            QMessageBox.warning(self, "No Session", "Please load a session first.")
            return

        dlg = ReconstructDialog(len(self.model.cat_list), self)
        if dlg.exec_() != QDialog.Accepted:
            return
        try:
            n_edges, thr, prune_thr = dlg.parameters()
        except Exception:
            QMessageBox.warning(self, "Invalid Input", "Enter valid numbers.")
            return

        app = QApplication.instance()
        if app:
            app.setOverrideCursor(Qt.WaitCursor)
        try:
            features = self.model.features
            oc_feats = self.model.openclip_features if len(self._model_names) > 1 else None
            plc_feats = self.model.places365_features if len(self._model_names) > 2 else None

            matrix, _ = temi_cluster(features, out_dim=n_edges, threshold=thr)
            oc_matrix = None
            if oc_feats is not None:
                oc_matrix, _ = temi_cluster(oc_feats, out_dim=n_edges, threshold=thr)
            plc_matrix = None
            if plc_feats is not None:
                plc_matrix, _ = temi_cluster(plc_feats, out_dim=n_edges, threshold=thr)

            empty_cols = np.where(matrix.sum(axis=0) == 0)[0]
            if len(empty_cols) > 0:
                matrix = np.delete(matrix, empty_cols, axis=1)
                QMessageBox.information(
                    self,
                    "Empty Hyperedges Removed",
                    f"{len(empty_cols)} empty hyperedges were removed after clustering."
                )
            if oc_matrix is not None:
                oc_empty = np.where(oc_matrix.sum(axis=0) == 0)[0]
                if len(oc_empty) > 0:
                    oc_matrix = np.delete(oc_matrix, oc_empty, axis=1)
            if plc_matrix is not None:
                plc_empty = np.where(plc_matrix.sum(axis=0) == 0)[0]
                if len(plc_empty) > 0:
                    plc_matrix = np.delete(plc_matrix, plc_empty, axis=1)
        except Exception as e:
            if app:
                app.restoreOverrideCursor()
            QMessageBox.critical(self, "Reconstruction Error", str(e))
            return
        if app:
            app.restoreOverrideCursor()

        df = pd.DataFrame(matrix.astype(int), columns=[f"edge_{i}" for i in range(matrix.shape[1])])

        try:
            self.model.layoutChanged.disconnect(self.regroup)
        except Exception:
            pass
        try:
            self.model.layoutChanged.disconnect(self._on_layout_changed)
        except Exception:
            pass
        try:
            self.model.hyperedgeModified.disconnect(self._on_model_hyperedge_modified)
        except Exception:
            pass

        self.model = SessionModel(
            self.model.im_list,
            df,
            features,
            self.model.h5_path,
            openclip_features=oc_feats,
            places365_features=plc_feats,
            thumbnail_data=self.model.thumbnail_data,
            thumbnails_are_embedded=self.model.thumbnails_are_embedded,
            metadata=self.model.metadata,
        )

        if oc_matrix is not None and oc_matrix.size:
            origin = self._model_names[1].lower()
            self.model.append_clustering_matrix(oc_matrix, origin=origin, prefix=origin)

        if plc_matrix is not None and plc_matrix.size:
            origin = self._model_names[2].lower()
            self.model.append_clustering_matrix(plc_matrix, origin=origin, prefix=origin)
        self.model.prune_similar_edges(prune_thr)

        self.model.layoutChanged.connect(self.regroup)
        self._overview_triplets = None
        self.model.layoutChanged.connect(self._on_layout_changed)
        self.model.hyperedgeModified.connect(self._on_model_hyperedge_modified)

        self.audio_table.clear()
        for name in self._model_names:
            self.audio_table.add_model(name, self.model)
        self.audio_table.set_use_full_images(True)
        self.thumb_toggle_act.setChecked(True)
        self.matrix_dock.set_model(self.model)
        self.spatial_dock.set_model(self.model)
        self.regroup()



    def _on_layout_changed(self):
        self._overview_triplets = None
        if self._layout_timer.isActive():
            self._layout_timer.stop()
        self._layout_timer.start(0)

    def _apply_layout_change(self):
        start_timer13 = time.perf_counter()        
        if hasattr(self, "spatial_dock") and not self._skip_next_layout:
            self.spatial_dock.set_model(self.model)
        self.regroup()
        self._skip_next_layout = False
        print('_apply_layout_change',time.perf_counter() - start_timer13)

    def toggle_full_images(self, flag: bool) -> None:
        self.audio_table.set_use_full_images(flag)

    def _on_model_hyperedge_modified(self, _name: str):
        self._skip_next_layout = True
        if self._skip_reset_timer.isActive():
            self._skip_reset_timer.stop()
        self._skip_reset_timer.start(100)

    def _on_item_changed(self, item: QStandardItem):
        if item.column() != 0 or item.hasChildren():
            return

        parent = item.parent()
        old_name, new_name = item.data(Qt.UserRole), item.text().strip()

        # --- Handle rename -------------------------------------------------
        if old_name != new_name:
            if not self.model.rename_edge(old_name, new_name):
                item.setText(old_name)
                return
            item.setData(new_name, Qt.UserRole)
            if hasattr(self, "groups"):
                for g, children in self.groups.items():
                    for idx, child in enumerate(children):
                        if child == old_name:
                            children[idx] = new_name
                            break
            if parent is not None:
                self._update_group_similarity(parent)

        # --- Handle visibility toggle -------------------------------------
        if item.isCheckable():
            visible = item.checkState() == Qt.Checked
            self.spatial_dock.set_edge_visible(new_name, visible)

    def _invalidate_similarity_column(self, name_item: QStandardItem):
        row = name_item.row()
        sim_item = name_item.parent().child(row, SIM_COL)
        sim_item.setData(None, Qt.UserRole); sim_item.setData("", Qt.DisplayRole)

    def _update_bus_images(self, names: list[str]):
        start_timer14 = time.perf_counter()
        
        if not self.model:
            self.bus.set_images([])
            return
        print('_update_bus_images0', time.perf_counter() - start_timer14)
        idxs = set()
        for name in names:
            if name in self.model.hyperedges:
                idxs.update(self.model.hyperedges.get(name, set()))
            elif hasattr(self, "groups") and name in self.groups:
                for child in self.groups[name]:
                    idxs.update(self.model.hyperedges.get(child, set()))
        print('_update_bus_images1', time.perf_counter() - start_timer14)
        self.bus.set_images(sorted(idxs))
        print('_update_bus_images2', time.perf_counter() - start_timer14)

    def _remember_last_edge(self, names: list[str]):
        if names:
            self._last_edge = names[0]

    def _show_legend(self, mapping: dict[str, str]):
        while self.legend_layout.count():
            item = self.legend_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for label, color in mapping.items():
            lab = QLabel(f"<span style='color:{color}'>■</span> {label}")
            self.legend_layout.addWidget(lab)
        self.legend_box.show()

    def _hide_legend(self):
        self.legend_box.hide()
    
    def _update_spatial_limits(self):
        try:
            img_val = int(self.limit_images_edit.text())
        except ValueError:
            img_val = 10
        try:
            edge_val = int(self.limit_edges_edit.text())
        except ValueError:
            edge_val = 10
        self.spatial_dock.set_image_limit(self.limit_images_cb.isChecked(), img_val)
        self.spatial_dock.set_intersection_limit(self.limit_edges_cb.isChecked(), edge_val)

    def choose_hidden_edges(self):
        if not self.model:
            return

        names = sorted(self.model.hyperedges)
        dialog = _MultiSelectDialog(names, self)
        hidden = self.spatial_dock.hidden_edges
        for i in range(dialog.list.count()):
            item = dialog.list.item(i)
            if item.text() in hidden:
                item.setSelected(True)
        if dialog.exec() != QDialog.Accepted:
            return

        to_hide = set(dialog.chosen())
        src_model = self._source_model()
        with QSignalBlocker(src_model):
            root = src_model.invisibleRootItem()
            for r in range(root.rowCount()):
                it = root.child(r, 0)
                if it.hasChildren():
                    for c in range(it.rowCount()):
                        leaf = it.child(c, 0)
                        if leaf.isCheckable():
                            leaf.setCheckState(Qt.Unchecked if leaf.text() in to_hide else Qt.Checked)
                else:
                    if it.isCheckable():
                        it.setCheckState(Qt.Unchecked if it.text() in to_hide else Qt.Checked)
        self.spatial_dock.set_hidden_edges(to_hide)

    # ------------------------------------------------------------------
    def color_edges_default(self):
        """Color hyperedge nodes with the session's stored colors."""
        if not self.model:
            return
        self.spatial_dock.update_colors(self.model.edge_colors)
        self.spatial_dock.hide_legend()
        self._hide_legend()

    def color_edges_by_status(self):
        """Color hyperedges based on their edit status."""
        if not self.model:
            return
        statuses = sorted({meta.get("status", "") for meta in self.model.status_map.values()})
        color_list = generate_n_colors(len(statuses))
        colors = {s: color_list[i % len(color_list)] for i, s in enumerate(statuses)}
        mapping = {name: colors[self.model.status_map[name]["status"]] for name in self.model.hyperedges}
        self.spatial_dock.update_colors(mapping)
        self.spatial_dock.show_legend(colors)
        self._show_legend(colors)

    def color_edges_by_origin(self):
        """Color hyperedges based on their origin."""
        if not self.model:
            return
        origins = sorted(set(self.model.edge_origins.values()))
        color_list = generate_n_colors(len(origins))
        colors = {o: color_list[i % len(color_list)] for i, o in enumerate(origins)}
        mapping = {name: colors[self.model.edge_origins.get(name, "") ] for name in self.model.hyperedges}
        self.spatial_dock.update_colors(mapping)
        self.spatial_dock.show_legend(colors)
        self._show_legend(colors)

    def color_edges_by_similarity(self):
        """Color hyperedges by similarity to the selected or last edge."""
        if not self.model:
            return
        sel = self.tree.selectionModel().selectedRows(0)
        ref = sel[0].data(Qt.DisplayRole) if sel else self._last_edge
        if not ref:
            return
        if not self._similarity_computed or self._similarity_ref != ref:
            self.compute_similarity(ref)
        sim_map = self.model.similarity_map(ref)
        if not sim_map:
            return
 
        max_v = max(sim_map.values())
        min_v = min(sim_map.values())
        denom = max(max_v - min_v, 1e-6)

        def interpolate_grey_to_red(norm):
            """Returns a QColor name from grey to red based on normalized similarity."""
            # Grey: (150, 150, 150), Red: (255, 0, 0)
            r = int(150 + norm * (255 - 150))
            g = int(150 - norm * 150)
            b = int(150 - norm * 150)
            return QColor(r, g, b).name()

        cmap = {}
        for name, val in sim_map.items():
            norm = (val - min_v) / denom
            col = interpolate_grey_to_red(norm)
            cmap[name] = col

        self.spatial_dock.update_colors(cmap)
        self.spatial_dock.hide_legend()
        self._hide_legend()

    def regroup(self):
        start_timer14 = time.perf_counter()
        if not self.model: 
            return
        thr = self.slider.value() / 100
        self.label.setText(f"Grouping threshold: {thr:.2f}")

        self.groups = rename_groups_sequentially(perform_hierarchical_grouping(self.model, thresh=thr))
        rows = build_row_data(self.groups, self.model)
        print('regroup', time.perf_counter() - start_timer14)
        headers = [
            "Name",
            "Images",
            "Status",
            "Origin",
            "Similarity",
            "Std. Dev.",
            "Intersection",
        ]
        model = build_qmodel(rows, headers)
        model.itemChanged.connect(self._on_item_changed)
        self.tree_proxy.setSourceModel(model)
        self.tree.setModel(self.tree_proxy)
        self.tree.selectionModel().selectionChanged.connect(self.tree._send_bus_update)
        self.tree.collapseAll()

        if hasattr(self, 'matrix_dock'): 
            self.matrix_dock.update_matrix()
        print('regroup1', time.perf_counter() - start_timer14)

    def _update_group_similarity(self, group_item: QStandardItem):
        vals = [v for v in (group_item.child(r, SIM_COL).data(Qt.UserRole) for r in range(group_item.rowCount())) if v is not None]
        parent = group_item.parent() or group_item.model().invisibleRootItem()
        sim_item = parent.child(group_item.row(), SIM_COL)

        if vals:
            mean_val = float(np.mean(vals))
            sim_item.setData(mean_val, Qt.UserRole); sim_item.setData(f"{mean_val:.{DECIMALS}f}", Qt.DisplayRole)
        else:
            sim_item.setData(None, Qt.UserRole); sim_item.setData("", Qt.DisplayRole)

# ---------- main -----------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    if SYSTEM_DARK_MODE:
        apply_dark_palette(app)
    win = MainWin()
    win.show()
    sys.exit(app.exec())