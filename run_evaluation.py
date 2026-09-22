"""Open Policy Evaluation in PyCharm. Nothing runs until Evaluate is pressed."""

from pathlib import Path

from PySide6 import QtWidgets

from tasks.whip import evaluation
from training.ui.evaluation_window import EvaluationWindow


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = EvaluationWindow(Path(__file__).resolve().parent, evaluation)
    window.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
