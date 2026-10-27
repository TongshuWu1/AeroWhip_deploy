"""One visual language for every AeroWhip window: Qt stylesheet and matplotlib defaults.

Colours follow the Okabe–Ito colour-blind-safe palette. Meaning is fixed across windows:
measured data is near-black, references are grey dashed, commands orange, model
predictions blue, errors vermillion; X/Y/Z components are blue/green/vermillion.
"""
import matplotlib

PALETTE = {
    'black': '#1f1f1f', 'grey': '#8c8c8c', 'blue': '#0072B2', 'orange': '#E69F00',
    'green': '#009E73', 'vermillion': '#D55E00', 'purple': '#CC79A7', 'sky': '#56B4E9',
    'yellow': '#F0E442',
}
# Semantic roles, shared by all plots.
ROLE = {
    'measured': PALETTE['black'], 'reference': PALETTE['grey'], 'command': PALETTE['orange'],
    'model': PALETTE['blue'], 'error': PALETTE['vermillion'], 'highlight': PALETTE['purple'],
    'evaluation': PALETTE['vermillion'], 'band': '#9fb7cc',
}
AXIS = (PALETTE['blue'], PALETTE['green'], PALETTE['vermillion'])     # X, Y, Z
CYCLE = (PALETTE['blue'], PALETTE['orange'], PALETTE['green'], PALETTE['vermillion'],
         PALETTE['purple'], PALETTE['sky'], PALETTE['black'])

FONT_PX = 13
_TOKENS = {
    False: dict(page='#f6f7f9', panel='#ffffff', text='#1f2933', muted='#5f6b76', border='#d3d8de',
                hover='#eef3f8', accent='#0072B2', accent_text='#ffffff', disabled='#a0a8b0',
                selection='#d6e8f5', error='#b23b1e', error_bg='#fbeae5', groove='#cfd6dd'),
    True: dict(page='#1b1f24', panel='#22272e', text='#dde3ea', muted='#93a0ad', border='#3a424c',
               hover='#2c333c', accent='#3d93d1', accent_text='#ffffff', disabled='#5d6670',
               selection='#244766', error='#ffb4a3', error_bg='#4a2a26', groove='#454e59'),
}


def stylesheet(dark=False):
    t = _TOKENS[bool(dark)]
    return f'''
QMainWindow, QDialog, QWidget#page, QWidget#trainingPage, QWidget#flightPage {{ background: {t['page']}; color: {t['text']}; }}
QWidget {{ font-size: {FONT_PX}px; }}
QLabel {{ color: {t['text']}; background: transparent; }}
QLabel#heading {{ font-size: 18px; font-weight: 600; }}
QLabel#muted, QLabel#subtle, QLabel#note {{ color: {t['muted']}; }}
QLabel#note {{ font-size: 12px; }}
QLabel#section {{ color: {t['muted']}; font-weight: 600; padding-top: 10px; }}
QLabel#value, QLabel#smallMetric {{ font-size: 17px; font-weight: 600; }}
QLabel#cardTitle {{ color: {t['muted']}; font-weight: 600; }}
QLabel#success {{ font-weight: 600; }}
QFrame#card {{ background: {t['panel']}; border: 1px solid {t['border']}; border-radius: 4px; }}
QLabel#error {{ color: {t['error']}; background: {t['error_bg']}; border-radius: 3px; padding: 6px; }}
QGroupBox {{ border: 1px solid {t['border']}; border-radius: 4px; margin-top: 10px; padding: 8px 6px 6px 6px; background: {t['panel']}; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 3px; color: {t['muted']}; font-weight: 600; }}
QPushButton, QToolButton, QComboBox, QLineEdit {{
    background: {t['panel']}; color: {t['text']}; border: 1px solid {t['border']}; border-radius: 3px; padding: 4px 9px; }}
QPushButton:hover, QToolButton:hover {{ background: {t['hover']}; }}
QPushButton:disabled, QToolButton:disabled, QComboBox:disabled {{ color: {t['disabled']}; }}
QPushButton#primary {{ background: {t['accent']}; color: {t['accent_text']}; border-color: {t['accent']}; font-weight: 600; }}
QPushButton#primary:disabled {{ background: {t['hover']}; color: {t['disabled']}; border-color: {t['border']}; }}
QComboBox QAbstractItemView {{ background: {t['panel']}; color: {t['text']}; selection-background-color: {t['selection']}; selection-color: {t['text']}; }}
QPlainTextEdit, QTextEdit, QTableWidget, QTreeWidget, QListWidget {{ background: {t['panel']}; color: {t['text']};
    border: 1px solid {t['border']}; selection-background-color: {t['selection']}; selection-color: {t['text']}; }}
QHeaderView::section {{ background: {t['page']}; color: {t['muted']}; border: 0; border-bottom: 1px solid {t['border']}; padding: 4px; font-weight: 600; }}
QTabWidget::pane {{ border: 1px solid {t['border']}; background: {t['panel']}; top: -1px; }}
QTabBar::tab {{ background: {t['page']}; color: {t['muted']}; padding: 6px 14px; border: 1px solid transparent; }}
QTabBar::tab:selected {{ background: {t['panel']}; color: {t['text']}; border: 1px solid {t['border']}; border-bottom-color: {t['panel']}; }}
QProgressBar {{ border: 1px solid {t['border']}; border-radius: 3px; background: {t['panel']}; color: {t['text']}; text-align: center; height: 16px; }}
QProgressBar::chunk {{ background: {t['selection']}; }}
QCheckBox {{ color: {t['text']}; spacing: 6px; }}
QSlider::groove:horizontal {{ background: {t['groove']}; height: 4px; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {t['accent']}; width: 12px; margin: -4px 0; border-radius: 6px; }}
QStatusBar {{ background: {t['page']}; color: {t['muted']}; border-top: 1px solid {t['border']}; }}
QStatusBar::item {{ border: 0; }}
QToolBar {{ background: {t['page']}; border: 0; border-bottom: 1px solid {t['border']}; spacing: 6px; padding: 5px; }}
QDockWidget {{ color: {t['muted']}; }}
QDockWidget::title {{ background: {t['page']}; padding: 6px 8px; border-bottom: 1px solid {t['border']}; }}
QScrollArea {{ border: 0; background: transparent; }}
QMenuBar, QMenu {{ background: {t['page']}; color: {t['text']}; }}
QMenu::item:selected {{ background: {t['selection']}; }}
QSplitter::handle {{ background: {t['border']}; }}
'''


def apply(widget, *, dark=False):
    """Style one top-level window (keeps windows independent of each other's theme)."""
    widget.setStyleSheet(stylesheet(dark))


def matplotlib_defaults(dark=False):
    t = _TOKENS[bool(dark)]
    return {
        'font.size': 9, 'axes.titlesize': 9.5, 'axes.titleweight': 'normal', 'axes.titlelocation': 'left',
        'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .7,
        'axes.edgecolor': t['muted'], 'axes.labelcolor': t['text'], 'axes.titlecolor': t['text'],
        'axes.facecolor': t['panel'], 'figure.facecolor': t['panel'], 'savefig.facecolor': t['panel'],
        'xtick.color': t['muted'], 'ytick.color': t['muted'], 'text.color': t['text'],
        'axes.grid': True, 'grid.color': t['border'], 'grid.linewidth': .5, 'grid.alpha': .8,
        'axes.prop_cycle': matplotlib.cycler(color=CYCLE), 'lines.linewidth': 1.2, 'lines.markersize': 3.5,
        'legend.frameon': True, 'legend.framealpha': .85, 'legend.edgecolor': 'none', 'legend.handlelength': 1.6, 'legend.borderaxespad': .2, 'legend.borderpad': .25, 'legend.columnspacing': 1.,
        'axes.xmargin': 0., 'axes.ymargin': .08, 'axes.autolimit_mode': 'data',
        'figure.constrained_layout.use': True, 'figure.constrained_layout.h_pad': .04,
        'figure.constrained_layout.w_pad': .04, 'image.cmap': 'viridis',
    }
