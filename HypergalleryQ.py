# prototype.py --------------------------------------------------------------
import os
os.environ["PYQTGRAPH_QT_LIB"] = "PyQt5"   # force pyqtgraph to PyQt5
os.environ.pop("QT_API", None)             # avoid other libs nudging Qt differently


import sys, uuid, numpy as np
import pandas as pd
import numpy as np
from utils.similarity import SIM_METRIC
from pathlib import Path
import io
import torch
from PIL import Image, ImageGrab, ImageOps
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
    QGridLayout,
    QComboBox
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
from utils.image_grid import ImageGridDock
from utils.overlap_list_dock import OverlapListDock
from utils.hyperedge_matrix2 import HyperedgeMatrixDock
# from utils.spatial_viewQv3 import SpatialViewQDock
from utils.spatial_viewQv4 import SpatialViewQDock, HyperedgeItem
from utils.feature_extraction import (
    Swinv2LargeFeatureExtractor,
    OpenClipFeatureExtractor,
    DenseNet161Places365FeatureExtractor,
)
from utils.file_utils import get_image_files
from utils.session_stats import show_session_stats
from utils.metadata_overview import show_metadata_overview
from clustering.temi_clustering import temi_cluster
from clustering.fuzzy_cmeans import fuzzy_cmeans_cluster
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
            f"Found {image_count} images.\n\n"
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

        layout.addWidget(QLabel("Clustering algorithm:"))
        self.alg_combo = QComboBox()
        self.alg_combo.addItems(["TEMI", "Fuzzy C-Means"])
        layout.addWidget(self.alg_combo)

        layout.addWidget(QLabel("Fuzzy C-Means m:"))
        self.m_edit = QLineEdit("1.1")
        layout.addWidget(self.m_edit)

        self.openclip_cb = QCheckBox("Include OpenCLIP features")
        self.openclip_cb.setChecked(True)
        layout.addWidget(self.openclip_cb)

        self.places_cb = QCheckBox("Include Places365 features")
        self.places_cb.setChecked(True)
        layout.addWidget(self.places_cb)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Start generating hypergraph")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def parameters(self) -> tuple[int, float, float, str, float, bool, bool]:
        return (
            int(self.edge_edit.text()),
            float(self.thr_edit.text()),
            float(self.jacc_edit.text()),
            self.alg_combo.currentText(),
            float(self.m_edit.text()),
            self.openclip_cb.isChecked(),
            self.places_cb.isChecked(),
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


# ---------- Qt helpers -----------------------------------------------------
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
        self.bus.edgesChanged.connect(self._update_bus_images)
        #self.bus.edgesChanged.connect(print)
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

        self.btn_sim = QPushButton("Similarity")
        self.btn_sim.clicked.connect(self.compute_similarity)

        self.btn_add_hyperedge = QPushButton("Add")
        self.btn_add_hyperedge.clicked.connect(self.on_add_hyperedge)

        self.btn_del_hyperedge = QPushButton("Delete")
        self.btn_del_hyperedge.clicked.connect(self.on_delete_hyperedge)

        self.btn_add_img = QPushButton("Add images")
        self.btn_add_img.clicked.connect(self.add_selection_to_hyperedge)

        self.btn_del_img = QPushButton("Remove images")
        self.btn_del_img.clicked.connect(self.on_remove_images)

        self.btn_rank = QPushButton("By selection")
        self.btn_rank.clicked.connect(self.rank_selected_images)

        self.btn_rank_edge = QPushButton("By hyperedge")
        self.btn_rank_edge.clicked.connect(self.rank_selected_hyperedge)

        self.btn_rank_file = QPushButton("By image file")
        self.btn_rank_file.clicked.connect(self.rank_image_file)

        self.btn_rank_clip = QPushButton("By clipboard")
        self.btn_rank_clip.clicked.connect(self.rank_clipboard_image)

        self.btn_overview = QPushButton("Images")
        self.btn_overview.clicked.connect(self.show_overview)

        self.text_query = QLineEdit()
        self.text_query.setPlaceholderText("Text query…")
        self.text_query.returnPressed.connect(self.rank_text_query)

        self.btn_rank_text = QPushButton("By text")
        self.btn_rank_text.clicked.connect(self.rank_text_query)

        self.btn_meta_overview = QPushButton("Metadata")
        self.btn_meta_overview.clicked.connect(self.show_metadata_overview)

        self.btn_color_default = QPushButton("Edge")
        self.btn_color_default.clicked.connect(self.color_edges_default)

        self.btn_color_status = QPushButton("Status")
        self.btn_color_status.clicked.connect(self.color_edges_by_status)

        self.btn_color_origin = QPushButton("Origin")
        self.btn_color_origin.clicked.connect(self.color_edges_by_origin)

        self.btn_color_similarity = QPushButton("Similarity")
        self.btn_color_similarity.clicked.connect(self.color_edges_by_similarity)

        self.btn_session_stats = QPushButton("Session stats")
        self.btn_session_stats.clicked.connect(self.show_session_stats)

        self.btn_manage_visibility = QPushButton("Visibility")
        self.btn_manage_visibility.clicked.connect(self.choose_hidden_edges)

        self.limit_images_cb = QCheckBox("Limit number of image nodes per hyperedge")
        self.limit_images_edit = QLineEdit("10")
        lim_img_row = QHBoxLayout(); lim_img_row.addWidget(self.limit_images_cb); lim_img_row.addWidget(self.limit_images_edit)
        lim_img_w = QWidget(); lim_img_w.setLayout(lim_img_row)

        self.limit_edges_cb = QCheckBox("Limit number of intersecting hyperedges")
        self.limit_edges_edit = QLineEdit("10")
        lim_edge_row = QHBoxLayout(); lim_edge_row.addWidget(self.limit_edges_cb); lim_edge_row.addWidget(self.limit_edges_edit)
        lim_edge_w = QWidget(); lim_edge_row.setContentsMargins(0,0,0,0); lim_img_row.setContentsMargins(0,0,0,0); lim_edge_w.setLayout(lim_edge_row)

        hyperedge_group = QGroupBox("Hyperedges")
        hyperedge_layout = QGridLayout()
        hyperedge_layout.addWidget(self.btn_add_hyperedge, 0, 0)
        hyperedge_layout.addWidget(self.btn_del_hyperedge, 0, 1)
        hyperedge_layout.addWidget(self.btn_add_img, 1, 0)
        hyperedge_layout.addWidget(self.btn_del_img, 1, 1)
        hyperedge_layout.addWidget(self.btn_manage_visibility, 2, 0, 1, 2)
        hyperedge_group.setLayout(hyperedge_layout)

        query_group = QGroupBox("Query")
        query_layout = QGridLayout()
        query_layout.addWidget(self.btn_rank, 0, 0)
        query_layout.addWidget(self.btn_rank_edge, 0, 1)
        query_layout.addWidget(self.btn_rank_file, 1, 0)
        query_layout.addWidget(self.btn_rank_clip, 1, 1)
        query_layout.addWidget(self.text_query, 2, 0, 1, 2)
        query_layout.addWidget(self.btn_rank_text, 3, 0, 1, 2)
        query_group.setLayout(query_layout)

        color_group = QGroupBox("Colorize")
        color_layout = QGridLayout()
        color_layout.addWidget(self.btn_color_default, 0, 0)
        color_layout.addWidget(self.btn_color_status, 0, 1)
        color_layout.addWidget(self.btn_color_origin, 1, 0)
        color_layout.addWidget(self.btn_color_similarity, 1, 1)
        color_group.setLayout(color_layout)

        overview_group = QGroupBox("Overview")
        overview_layout = QGridLayout()
        overview_layout.addWidget(self.btn_sim, 0, 0, 1, 2)
        overview_layout.addWidget(self.btn_overview, 1, 0)
        overview_layout.addWidget(self.btn_meta_overview, 1, 1)
        overview_layout.addWidget(self.btn_session_stats, 2, 0, 1, 2)
        overview_group.setLayout(overview_layout)

        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout()
        options_layout.addWidget(lim_img_w)
        options_layout.addWidget(lim_edge_w)


        self.image_grid = ImageGridDock(self.bus, self)

        options_layout.addWidget(self.image_grid.hide_selected_cb)
        options_layout.addWidget(self.image_grid.hide_modified_cb)


        options_group.setLayout(options_layout)

        toolbar_layout.addWidget(hyperedge_group)
        toolbar_layout.addWidget(query_group)
        toolbar_layout.addWidget(color_group)
        toolbar_layout.addWidget(overview_group)
        toolbar_layout.addWidget(options_group)



        self.legend_box = QGroupBox("Legend")
        self.legend_layout = QVBoxLayout(self.legend_box)
        self.legend_layout.setContentsMargins(4, 4, 4, 4)
        self.legend_box.hide()
        toolbar_layout.addWidget(self.legend_box)

        self.overlap_dock = OverlapListDock(self.bus, self.image_grid, self)
        self.overlap_dock.setObjectName("OverlapListDock")        

        toolbar_layout.addStretch()
        self.toolbar_dock.setWidget(toolbar_container)

        self.image_grid.setObjectName("ImageGridDock") # Use object name for clarity
        self.image_grid.labelDoubleClicked.connect(lambda name: self.bus.set_edges([name]))
        self.spatial_dock = SpatialViewQDock(self.bus, self)
        self.spatial_dock.setObjectName("SpatialViewDock")
        self.limit_images_cb.toggled.connect(self._update_spatial_limits)
        self.limit_images_edit.editingFinished.connect(self._update_spatial_limits)
        self.limit_edges_cb.toggled.connect(self._update_spatial_limits)
        self.limit_edges_edit.editingFinished.connect(self._update_spatial_limits)
        self._update_spatial_limits()

        self.matrix_dock = HyperedgeMatrixDock(self.bus, self)
        self.matrix_dock.setObjectName("HyperedgeMatrixDock")

        self.slider = QSlider(Qt.Horizontal, minimum=50, maximum=100, singleStep=5, value=int(THRESHOLD_DEFAULT * 100))
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        toolbar_layout.insertWidget(0, self.label)
        toolbar_layout.insertWidget(1, self.slider)


        # DOCK LAYOUT 
        self.addDockWidget(Qt.LeftDockWidgetArea, self.tree_dock)

        self.addDockWidget(Qt.LeftDockWidgetArea, self.toolbar_dock)

        self.addDockWidget(Qt.RightDockWidgetArea, self.image_grid)
        self.addDockWidget(Qt.RightDockWidgetArea, self.overlap_dock)
        self.splitDockWidget(self.image_grid, self.overlap_dock, Qt.Horizontal)
        self.addDockWidget(Qt.RightDockWidgetArea, self.spatial_dock)

        self.splitDockWidget(self.spatial_dock, self.matrix_dock, Qt.Horizontal)

        self.resizeDocks([self.tree_dock, self.image_grid], [300, 850], Qt.Horizontal)
        self.resizeDocks([self.image_grid, self.overlap_dock], [700, 200], Qt.Horizontal)
        self.resizeDocks([self.tree_dock, self.toolbar_dock], [550, 250], Qt.Vertical)
        self.resizeDocks([self.spatial_dock, self.matrix_dock], [450, 450], Qt.Horizontal)


        #  MENU AND STATE 
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
        # self.menuBar().addMenu("&File").addAction(open_act)

        self.model = None

        self.temi_results = {}


        self.slider.valueChanged.connect(self.regroup)

        files = get_h5_files_in_directory()
        if files:
            self.load_session(Path(DATA_DIRECTORY) / files[0])


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

    def on_remove_images(self):
        if not self.model:
            QMessageBox.warning(self, "No Session", "Please load a session first.")
            return

        if not self.image_grid.view.model():
            QMessageBox.warning(self, "No Images", "No images are currently displayed.")
            return

        sel_indexes = self.image_grid.view.selectionModel().selectedIndexes()
        if not sel_indexes:
            QMessageBox.warning(self, "No Selection", "Select images to remove first.")
            return

        model = self.image_grid.view.model()
        img_idxs = [model._indexes[i.row()] for i in sel_indexes]

        possible_edges = sorted(
            {e for idx in img_idxs for e in self.model.image_mapping.get(idx, set())}
        )
        if not possible_edges:
            QMessageBox.information(
                self,
                "Not in Hyperedge",
                "Selected images are not assigned to any hyperedge.",
            )
            return

        dialog = _MultiSelectDialog(possible_edges, self)
        if dialog.exec() != QDialog.Accepted:
            return
        edges = dialog.chosen()
        if not edges:
            return

        self.model.remove_images_from_edges(img_idxs, edges)
        # refresh displayed images of current selection
        self._update_bus_images(self.image_grid._selected_edges)



    def rank_selected_images(self):
        """Rank all images by similarity to the currently selected images."""
        if not self.model:
            QMessageBox.warning(self, "No Session", "Please load a session first.")
            return

        model = self.image_grid.view.model()
        sel = self.image_grid.view.selectionModel().selectedRows() if model else []
        if not sel:
            QMessageBox.information(self, "No Selection", "Select images in the grid first.")
            return

        sel_idxs = [model._indexes[i.row()] for i in sel]
        feats = self.model.features
        ref = feats[sel_idxs].mean(axis=0, keepdims=True)
        sims = SIM_METRIC(ref, feats)[0]
        ranked = np.argsort(sims)[::-1]
        ranked = [i for i in ranked if i not in sel_idxs][:500]
        final_idxs = sel_idxs + ranked
        self.image_grid.update_images(
            final_idxs, highlight=sel_idxs, sort=False, query=True
        )

    def rank_selected_hyperedge(self):
        """Rank all images by similarity to the selected hyperedge."""
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
        if name not in self.model.hyperedges:
            QMessageBox.information(self, "Unknown Hyperedge", "Selected item is not a hyperedge.")
            return

        ref = self.model.hyperedge_avg_features.get(name)
        feats = self.model.features
        sims = SIM_METRIC(ref.reshape(1, -1), feats)[0]
        ranked = np.argsort(sims)[::-1]
        exclude = self.model.hyperedges[name]
        ranked = [i for i in ranked if i not in exclude][:500]
        self.image_grid.update_images(ranked, sort=False, query=True)

    def rank_image_file(self):
        """Rank all session images against a user chosen image file."""
        if not self.model:
            QMessageBox.warning(self, "No Session", "Please load a session first.")
            return

        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select image file",
            str(DATA_DIRECTORY),
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tiff);;All Files (*)",
        )
        if not file:
            return

        try:
            if not hasattr(self, "_ext_feature_extractor"):
                self._ext_feature_extractor = Swinv2LargeFeatureExtractor(batch_size=1)
            vec = self._ext_feature_extractor.extract_features([file])[0]
        except Exception as e:
            QMessageBox.critical(self, "Feature Error", str(e))
            return

        feats = self.model.features
        sims = SIM_METRIC(vec.reshape(1, -1), feats)[0]
        ranked = np.argsort(sims)[::-1][:500]
        self.image_grid.update_images(list(ranked), sort=False, query=True)

    def rank_clipboard_image(self):
        """Rank images by similarity to the image currently in the clipboard."""
        if not self.model:
            QMessageBox.warning(self, "No Session", "Please load a session first.")
            return
        try:
            clip_img = ImageGrab.grabclipboard()
        except Exception as e:
            QMessageBox.critical(self, "Clipboard Error", str(e))
            return

        if clip_img is None:
            QMessageBox.information(self, "No Image", "No image in clipboard.")
            return

        if isinstance(clip_img, list):
            try:
                clip_img = ImageOps.exif_transpose(Image.open(clip_img[0]))
            except Exception:
                QMessageBox.information(self, "No Image", "Clipboard does not contain an image.")
                return

        clip_img = ImageOps.exif_transpose(clip_img)
        if clip_img.mode == "RGBA":
            clip_img = clip_img.convert("RGB")

        if self._clip_extractor is None:
            self._clip_extractor = Swinv2LargeFeatureExtractor(batch_size=1)

        tensor = self._clip_extractor.transform(clip_img).unsqueeze(0).to(self._clip_extractor.device)
        with torch.no_grad():
            feat = self._clip_extractor.model(tensor).cpu().numpy()[0]

        feats = self.model.features
        sims = SIM_METRIC(feat.reshape(1, -1), feats)[0]
        ranked = np.argsort(sims)[::-1][:500]
        self.image_grid.update_images(ranked, sort=False, query=True)


    def rank_text_query(self):
        """Rank images by similarity to a text query using OpenCLIP."""
        if not self.model:
            QMessageBox.warning(self, "No Session", "Please load a session first.")
            return
        text = self.text_query.text().strip()
        if not text:
            QMessageBox.information(self, "No Text", "Enter a text query first.")
            return
        if self.model.openclip_features is None:
            QMessageBox.warning(self, "No Features", "Session lacks OpenCLIP features.")
            return
        if self._openclip_extractor is None:
            self._openclip_extractor = OpenClipFeatureExtractor(batch_size=1)
        vec = self._openclip_extractor.encode_text([text])[0]
        feats = self.model.openclip_features
        sims = SIM_METRIC(vec.reshape(1, -1), feats)[0]
        ranked = np.argsort(sims)[::-1][:500]
        self.image_grid.update_images(list(ranked), sort=False, query=True)

    def show_overview(self):
        """Display triplet overview on the image grid."""
        if not self.model:
            QMessageBox.warning(self, "No Session", "Please load a session first.")
            return
        if self._overview_triplets is None:
            self._overview_triplets = self._compute_overview_triplets()
        self.image_grid.show_overview(self._overview_triplets, self.model)

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


    def add_selection_to_hyperedge(self):
        """Add currently selected images in the grid to a chosen hyperedge."""
        if not self.model:
            QMessageBox.warning(self, "No Session", "Please load a session first.")
            return

        model = self.image_grid.view.model()
        sel = self.image_grid.view.selectionModel().selectedRows() if model else []
        if not sel:
            QMessageBox.information(self, "No Selection", "Select images in the grid first.")
            return

        img_idxs = [model._indexes[i.row()] for i in sel]

        dialog = HyperedgeSelectDialog(list(self.model.hyperedges), self)
        if dialog.exec_() != QDialog.Accepted:
            return
        name = dialog.selected_name()
        if not name:
            QMessageBox.information(self, "No Hyperedge", "No hyperedge selected.")
            return

        self.model.add_images_to_hyperedge(name, img_idxs)


    def new_session(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select image folder", str(DATA_DIRECTORY)
        )
        if not directory:
            return

        files = get_image_files(directory)
        if not files:
            QMessageBox.information(
                self, "No Images", "No supported image files found."
            )
            return

        dlg = NewSessionDialog(len(files), self)
        if dlg.exec_() != QDialog.Accepted:
            return
        try:
            n_edges, thr, prune_thr, algo, m_val, use_oc, use_plc = dlg.parameters()
        except Exception:
            QMessageBox.warning(self, "Invalid Input", "Enter valid numbers.")
            return

        app = QApplication.instance()
        if app:
            app.setOverrideCursor(Qt.WaitCursor)
        try:
            extractor = Swinv2LargeFeatureExtractor()
            features = extractor.extract_features(files)
            oc_features = None
            plc_features = None
            if use_oc:
                oc_extractor = OpenClipFeatureExtractor()
                oc_features = oc_extractor.extract_features(files)
            if use_plc:
                plc_extractor = DenseNet161Places365FeatureExtractor()
                plc_features = plc_extractor.extract_features(files)
            if algo == "Fuzzy C-Means":
                matrix, _ = fuzzy_cmeans_cluster(features, n_edges, thr, m_val)
                oc_matrix = np.array([])
                if oc_features is not None:
                    oc_matrix, _ = fuzzy_cmeans_cluster(oc_features, n_edges, thr, m_val)
                plc_matrix = np.array([])
                if plc_features is not None:
                    plc_matrix, _ = fuzzy_cmeans_cluster(plc_features, n_edges, thr, m_val)
            else:
                matrix, _ = temi_cluster(features, out_dim=n_edges, threshold=thr)
                oc_matrix = np.array([])
                if oc_features is not None:
                    oc_matrix, _ = temi_cluster(oc_features, out_dim=n_edges, threshold=thr)
                plc_matrix = np.array([])
                if plc_features is not None:
                    plc_matrix, _ = temi_cluster(plc_features, out_dim=n_edges, threshold=thr)
            empty_cols = np.where(matrix.sum(axis=0) == 0)[0]
            if len(empty_cols) > 0:
                matrix = np.delete(matrix, empty_cols, axis=1)
                QMessageBox.information(
                    self,
                    "Empty Hyperedges Removed",
                    f"{len(empty_cols)} empty hyperedges were removed after clustering."
                )
            if oc_matrix.size:
                oc_empty = np.where(oc_matrix.sum(axis=0) == 0)[0]
                if len(oc_empty) > 0:
                    oc_matrix = np.delete(oc_matrix, oc_empty, axis=1)
            if plc_matrix.size:
                plc_empty = np.where(plc_matrix.sum(axis=0) == 0)[0]
                if len(plc_empty) > 0:
                    plc_matrix = np.delete(plc_matrix, plc_empty, axis=1)
        except Exception as e:
            if app:
                app.restoreOverrideCursor()
            QMessageBox.critical(self, "Generation Error", str(e))
            return
        if app:
            app.restoreOverrideCursor()

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
        )
        if oc_matrix.size:
            self.model.append_clustering_matrix(oc_matrix, origin="openclip", prefix="clip")
        if plc_matrix.size:
            self.model.append_clustering_matrix(plc_matrix, origin="places365", prefix="plc365")
        self.model.prune_similar_edges(prune_thr)


        self.model.layoutChanged.connect(self.regroup)
        self._overview_triplets = None
        self.model.layoutChanged.connect(self._on_layout_changed)
        self.model.hyperedgeModified.connect(self._on_model_hyperedge_modified)

        self.image_grid.set_model(self.model)
        self.image_grid.set_use_full_images(True)
        self.thumb_toggle_act.setChecked(True)
        self.overlap_dock.set_model(self.model)        
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
            # --- Optional feature extraction ---------------------------------
            if self.model.openclip_features is None:
                if QMessageBox.question(
                    self,
                    "Missing OpenCLIP Features",
                    "OpenCLIP features are absent in this session.\nGenerate them now?",
                ) == QMessageBox.Yes:
                    oc_extractor = OpenClipFeatureExtractor()
                    self.model.openclip_features = oc_extractor.extract_features(
                        self.model.im_list
                    )
            if self.model.places365_features is None:
                if QMessageBox.question(
                    self,
                    "Missing Places365 Features",
                    "Places365 features are absent in this session.\nGenerate them now?",
                ) == QMessageBox.Yes:
                    plc_extractor = DenseNet161Places365FeatureExtractor()
                    self.model.places365_features = plc_extractor.extract_features(
                        self.model.im_list
                    )

            # --- Optional hyperedge generation -------------------------------
            if (
                self.model.openclip_features is not None
                and all(orig != "openclip" for orig in self.model.edge_origins.values())
            ):
                if QMessageBox.question(
                    self,
                    "Generate OpenCLIP Hyperedges",
                    "No OpenCLIP hyperedges found.\nGenerate them now?",
                ) == QMessageBox.Yes:
                    n_edges = len(self.model.cat_list)
                    oc_matrix, _ = temi_cluster(
                        self.model.openclip_features,
                        out_dim=n_edges,
                        threshold=THRESHOLD_DEFAULT,
                    )
                    oc_empty = np.where(oc_matrix.sum(axis=0) == 0)[0]
                    if len(oc_empty) > 0:
                        oc_matrix = np.delete(oc_matrix, oc_empty, axis=1)
                    self.model.append_clustering_matrix(
                        oc_matrix, origin="openclip", prefix="clip"
                    )
            if (
                self.model.places365_features is not None
                and all(orig != "places365" for orig in self.model.edge_origins.values())
            ):
                if QMessageBox.question(
                    self,
                    "Generate Places365 Hyperedges",
                    "No Places365 hyperedges found.\nGenerate them now?",
                ) == QMessageBox.Yes:
                    n_edges = len(self.model.cat_list)
                    plc_matrix, _ = temi_cluster(
                        self.model.places365_features,
                        out_dim=n_edges,
                        threshold=THRESHOLD_DEFAULT,
                    )
                    plc_empty = np.where(plc_matrix.sum(axis=0) == 0)[0]
                    if len(plc_empty) > 0:
                        plc_matrix = np.delete(plc_matrix, plc_empty, axis=1)
                    self.model.append_clustering_matrix(
                        plc_matrix, origin="places365", prefix="plc365"
                    )

            self.model.layoutChanged.connect(self.regroup)
            self._overview_triplets = None
            self.model.layoutChanged.connect(self._on_layout_changed)
            self.model.hyperedgeModified.connect(self._on_model_hyperedge_modified)            
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))
            return

        self.image_grid.set_model(self.model)
        if self.model.thumbnail_data:
            self.image_grid.set_use_full_images(False)
            self.thumb_toggle_act.setChecked(False)
        else:
            self.image_grid.set_use_full_images(True)
            self.thumb_toggle_act.setChecked(True)
        self.overlap_dock.set_model(self.model)            
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
            oc_feats = self.model.openclip_features
            plc_feats = self.model.places365_features

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
            self.model.append_clustering_matrix(oc_matrix, origin="openclip", prefix="clip")
        if plc_matrix is not None and plc_matrix.size:
            self.model.append_clustering_matrix(plc_matrix, origin="places365", prefix="plc365")
        self.model.prune_similar_edges(prune_thr)

        self.model.layoutChanged.connect(self.regroup)
        self._overview_triplets = None
        self.model.layoutChanged.connect(self._on_layout_changed)
        self.model.hyperedgeModified.connect(self._on_model_hyperedge_modified)

        self.image_grid.set_model(self.model)
        if self.model.thumbnail_data:
            self.image_grid.set_use_full_images(False)
            self.thumb_toggle_act.setChecked(False)
        else:
            self.image_grid.set_use_full_images(True)
            self.thumb_toggle_act.setChecked(True)
        self.overlap_dock.set_model(self.model)
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
        self.image_grid.set_use_full_images(flag)

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