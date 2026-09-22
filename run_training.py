"""Open the Training window in PyCharm. Nothing runs until Start is pressed."""

from pathlib import Path

from PySide6 import QtWidgets

from tasks.whip.experiment import load_experiment
from tasks.whip.resume import inspect_resume
from training.ui.window import TrainingWindow


CONFIGURATION = Path(__file__).resolve().parent/'configs/experiments/whip_stage1_directed.toml'
RESUME_CHECKPOINT = None
TOTAL_UPDATES = 100


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    root = Path(__file__).resolve().parent
    window = TrainingWindow(root, CONFIGURATION, load_experiment,
                            runs_directory=root/'runs/whip_single_scenario', resume_inspector=inspect_resume,
                            resume_checkpoint=RESUME_CHECKPOINT, total_updates=TOTAL_UPDATES,
                            additional_runs_directories=(root/'runs/whip_stage1',))
    window.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
