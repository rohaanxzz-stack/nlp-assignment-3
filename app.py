"""
Twitter Sentiment Analysis - BERT + PyQt5 GUI
Assignment 03 - NLP Lab

An enterprise-grade, modern async interface featuring multi-threaded inference,
individual sentiment probability breakdowns, and structured Fluent Dark aesthetics.
"""

import sys
import os
import traceback
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem,
    QHeaderView, QTextEdit, QFrame, QProgressBar, QMessageBox,
    QSplitter, QScrollArea
)

# Global styling configuration 
COLOR_MAP = {
    "Positive": "#10b981",    # Emerald
    "Negative": "#ef4444",    # Rose
    "Neutral": "#f59e0b",     # Amber
    "Irrelevant": "#6b7280",  # Slate Gray
    "Primary": "#3b82f6",     # Modern Accent Blue
}

DEFAULT_LABEL_MAP = {0: "Irrelevant", 1: "Negative", 2: "Neutral", 3: "Positive"}


# --------------------------------------------------------------------------
# Multi-Threaded Business Logic Layer
# --------------------------------------------------------------------------

class ModelLoaderThread(QThread):
    finished_ok = pyqtSignal(object, object, dict)
    failed = pyqtSignal(str)

    def __init__(self, model_dir):
        super().__init__()
        self.model_dir = model_dir

    def run(self):
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            model.eval()

            id2label = getattr(model.config, "id2label", None)
            if id2label and len(id2label) > 0:
                label_map = {int(k): str(v).capitalize() for k, v in id2label.items()}
            else:
                label_map = DEFAULT_LABEL_MAP

            self.finished_ok.emit(model, tokenizer, label_map)
        except Exception as e:
            self.failed.emit(f"{str(e)}\n\n{traceback.format_exc()}")


class PredictThread(QThread):
    finished_ok = pyqtSignal(str, float, list)
    failed = pyqtSignal(str)

    def __init__(self, model, tokenizer, label_map, text):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.text = text

    def run(self):
        try:
            inputs = self.tokenizer(
                self.text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128,
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=1).squeeze(0)

            pred_id = int(torch.argmax(probs).item())
            confidence = float(probs[pred_id].item()) * 100
            label = self.label_map.get(pred_id, str(pred_id))
            
            all_probs = [
                (self.label_map.get(i, str(i)), float(p.item()) * 100)
                for i, p in enumerate(probs)
            ]
            self.finished_ok.emit(label, confidence, all_probs)
        except Exception as e:
            self.failed.emit(f"{str(e)}\n\n{traceback.format_exc()}")


# --------------------------------------------------------------------------
# UI Components & Main Application Context
# --------------------------------------------------------------------------

class SentimentApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.label_map = DEFAULT_LABEL_MAP
        self.df = None
        
        # Thread references to protect against Garbage Collection lifecycle crashes
        self.loader_thread = None
        self.predict_thread = None

        self.setWindowTitle("Cognitive Sentiment Analyzer Engine")
        self.resize(1300, 820)
        self.setMinimumSize(1150, 720)

        self._build_ui()
        self._apply_styles()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(18)

        # Header 
        header = QFrame()
        header.setObjectName("headerCard")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(24, 20, 24, 20)

        title = QLabel("Twitter Sentiment Intelligence System")
        title.setObjectName("titleLabel")
        subtitle = QLabel("Harnessing fine-tuned Bidirectional Encoder Representations from Transformers (BERT) for real-time text sequences analysis.")
        subtitle.setObjectName("subtitleLabel")

        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        root.addWidget(header)

        # Controls Toolbar
        toolbar = QFrame()
        toolbar.setObjectName("card")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(20, 14, 20, 14)
        toolbar_layout.setSpacing(14)

        self.btn_load_dataset = QPushButton("📂  Import Dataset CSV")
        self.btn_load_model = QPushButton("🤖  Initialize BERT Model")
        self.btn_predict_selected = QPushButton("🔮  Analyze Selected Sequence")

        for b in (self.btn_load_dataset, self.btn_load_model, self.btn_predict_selected):
            b.setCursor(Qt.PointingHandCursor)
            b.setMinimumHeight(44)

        self.btn_load_dataset.clicked.connect(self.load_dataset)
        self.btn_load_model.clicked.connect(self.load_model)
        self.btn_predict_selected.clicked.connect(self.predict_selected_row)
        self.btn_predict_selected.setEnabled(False)

        self.status_label = QLabel("🔴 Core Model Offline  •  🔴 Data Stream Disconnected")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        toolbar_layout.addWidget(self.btn_load_dataset)
        toolbar_layout.addWidget(self.btn_load_model)
        toolbar_layout.addWidget(self.btn_predict_selected)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self.status_label)
        root.addWidget(toolbar)

        # Primary Workspaces Splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # Left Container Workspace (Table view)
        left_card = QFrame()
        left_card.setObjectName("card")
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(20, 20, 20, 20)

        left_title = QLabel("Target Corpus Ingestion View")
        left_title.setObjectName("cardTitle")
        left_layout.addWidget(left_title)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Index", "Tweet Content Frame", "Ground Truth"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.cellClicked.connect(self.on_row_clicked)
        left_layout.addWidget(self.table)

        # Right Container Workspace (Analysis controls + Analytics breakdown)
        right_card = QFrame()
        right_card.setObjectName("card")
        right_layout = QVBoxLayout(right_card)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(14)

        right_title = QLabel("Analytical Workbench")
        right_title.setObjectName("cardTitle")
        right_layout.addWidget(right_title)

        sel_label = QLabel("Sequence Selected for Assessment")
        sel_label.setObjectName("fieldLabel")
        self.selected_text = QTextEdit()
        self.selected_text.setReadOnly(True)
        self.selected_text.setFixedHeight(75)
        self.selected_text.setPlaceholderText("Awaiting sequence click configuration from data engine...")

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setObjectName("divider")

        pred_title = QLabel("Model Output Prediction Class")
        pred_title.setObjectName("fieldLabel")
        self.prediction_box = QLabel("AWAITING DATA Pipeline")
        self.prediction_box.setObjectName("predictionBox")
        self.prediction_box.setAlignment(Qt.AlignCenter)
        self.prediction_box.setMinimumHeight(60)

        conf_title = QLabel("Primary Class Confidence Value")
        conf_title.setObjectName("fieldLabel")
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setFormat("%p%")

        # Container Area for Advanced Fine-Grained Probabilities Metrics
        distribution_title = QLabel("Metric Probability Space Distribution")
        distribution_title.setObjectName("fieldLabel")
        
        self.metrics_scroll = QScrollArea()
        self.metrics_scroll.setWidgetResizable(True)
        self.metrics_scroll.setObjectName("metricsArea")
        self.metrics_scroll.setFrameShape(QFrame.NoFrame)
        
        self.metrics_container = QWidget()
        self.metrics_layout = QVBoxLayout(self.metrics_container)
        self.metrics_layout.setContentsMargins(4, 4, 4, 4)
        self.metrics_layout.setSpacing(8)
        self.metrics_scroll.setWidget(self.metrics_container)

        right_layout.addWidget(sel_label)
        right_layout.addWidget(self.selected_text)
        right_layout.addWidget(divider)
        right_layout.addWidget(pred_title)
        right_layout.addWidget(self.prediction_box)
        right_layout.addWidget(conf_title)
        right_layout.addWidget(self.confidence_bar)
        right_layout.addWidget(distribution_title)
        right_layout.addWidget(self.metrics_scroll)

        # Custom Manual Sandbox Entry Area
        manual_title = QLabel("Ad-hoc Evaluation Sandbox")
        manual_title.setObjectName("fieldLabel")
        self.manual_input = QTextEdit()
        self.manual_input.setFixedHeight(70)
        self.manual_input.setPlaceholderText("Type a custom text string configuration to execute an inference pass...")

        self.btn_predict_manual = QPushButton("✨  Evaluate Custom Sequence")
        self.btn_predict_manual.setCursor(Qt.PointingHandCursor)
        self.btn_predict_manual.setMinimumHeight(44)
        self.btn_predict_manual.clicked.connect(self.predict_manual_text)
        self.btn_predict_manual.setEnabled(False)

        right_layout.addWidget(manual_title)
        right_layout.addWidget(self.manual_input)
        right_layout.addWidget(self.btn_predict_manual)

        splitter.addWidget(left_card)
        splitter.addWidget(right_card)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 3)
        root.addWidget(splitter, stretch=1)

        # Standard Core Status bar setup
        self.statusBar().showMessage("System core initialization sequence finalized.")

    def _apply_styles(self):
        # A professional Dark Arc UI Sheet scheme
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: #0b0f19;
            }}
            QLabel, QWidget {{
                color: #f1f5f9;
                font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
            }}
            #headerCard {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1e293b, stop:1 #0f172a);
                border-radius: 12px;
                border: 1px solid #334155;
            }}
            #titleLabel {{
                font-size: 26px;
                font-weight: 700;
                color: #ffffff;
                letter-spacing: -0.5px;
            }}
            #subtitleLabel {{
                font-size: 13px;
                color: #94a3b8;
                margin-top: 4px;
            }}
            #card {{
                background-color: #111827;
                border-radius: 12px;
                border: 1px solid #1f2937;
            }}
            #cardTitle {{
                font-size: 16px;
                font-weight: 600;
                color: #f3f4f6;
                padding-bottom: 4px;
            }}
            #fieldLabel {{
                font-size: 11px;
                font-weight: 700;
                color: #64748b;
                text-transform: uppercase;
                letter-spacing: 0.75px;
                margin-top: 4px;
            }}
            #statusLabel {{
                font-size: 13px;
                color: #94a3b8;
                font-weight: 500;
            }}
            #divider {{
                background-color: #1f2937;
                max-height: 1px;
                border: none;
            }}
            #predictionBox {{
                background-color: #1f2937;
                border: 1px solid #374151;
                border-radius: 8px;
                font-size: 18px;
                font-weight: 700;
                color: #9ca3af;
                letter-spacing: 1px;
            }}
            #metricsArea {{
                background-color: transparent;
            }}
            QPushButton {{
                background-color: {COLOR_MAP['Primary']};
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 8px 18px;
                font-weight: 600;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background-color: #2563eb;
            }}
            QPushButton:pressed {{
                background-color: #1d4ed8;
            }}
            QPushButton:disabled {{
                background-color: #1f2937;
                color: #4b5563;
            }}
            QTableWidget {{
                background-color: #030712;
                gridline-color: #111827;
                border: 1px solid #1f2937;
                border-radius: 8px;
                selection-background-color: #1e3a8a;
                selection-color: #ffffff;
                font-size: 13px;
            }}
            QTableWidget::item {{
                padding: 8px;
            }}
            QHeaderView::section {{
                background-color: #1f2937;
                color: #f3f4f6;
                padding: 8px;
                border: none;
                font-weight: 600;
                font-size: 12px;
            }}
            QTextEdit {{
                background-color: #030712;
                border: 1px solid #1f2937;
                border-radius: 6px;
                padding: 8px;
                color: #f9fafb;
                font-size: 13px;
            }}
            QTextEdit:focus {{
                border: 1px solid {COLOR_MAP['Primary']};
            }}
            QProgressBar {{
                background-color: #030712;
                border: 1px solid #1f2937;
                border-radius: 6px;
                text-align: center;
                color: #ffffff;
                font-weight: bold;
                height: 24px;
            }}
            QProgressBar::chunk {{
                background-color: {COLOR_MAP['Primary']};
                border-radius: 4px;
            }}
            QStatusBar {{
                background-color: #030712;
                color: #6b7280;
                font-size: 12px;
                border-top: 1px solid #1f2937;
            }}
        """)

    # --------------------------------------------------------------------------
    # Business Ingestion Logic & Async Controller Actions
    # --------------------------------------------------------------------------

    def load_dataset(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Twitter Sentiment CSV Corpus", "", "CSV Metrics Data (*.csv)"
        )
        if not path:
            return

        try:
            df = pd.read_csv(path)
        except Exception as e:
            QMessageBox.critical(self, "Data Ingestion Failure", f"Failed to parse source file architecture:\n{str(e)}")
            return

        text_col = self._find_column(df, ["text", "tweet", "tweet_text", "sentence"])
        label_col = self._find_column(df, ["sentiment", "label", "target", "class"])

        if text_col is None:
            QMessageBox.critical(
                self, "Incompatible Matrix Structure",
                "Unable to identify critical feature sequence arrays.\n"
                "Ensure standard formatting header is present (e.g., text, tweet)."
            )
            return

        self.df = df
        self.text_col = text_col
        self.label_col = label_col

        self.table.setRowCount(0)
        self.table.setSortingEnabled(False)
        
        # Performance Bulk Loading Configuration
        for i, row in df.iterrows():
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(str(i + 1)))
            self.table.setItem(r, 1, QTableWidgetItem(str(row[text_col])))
            actual = str(row[label_col]) if label_col else "Not Provided"
            self.table.setItem(r, 2, QTableWidgetItem(actual))
            
        self._update_status()
        self.statusBar().showMessage(f"Ingested {len(df)} sequences mapped successfully to the display buffer.")

    @staticmethod
    def _find_column(df, candidates):
        cols_lower = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand in cols_lower:
                return cols_lower[cand]
        return None

    def load_model(self):
        model_dir = QFileDialog.getExistingDirectory(self, "Locate Pre-trained Save Directory")
        if not model_dir:
            return

        self.btn_load_model.setEnabled(False)
        self.statusBar().showMessage("Initializing neural tensor mappings...")

        self.loader_thread = ModelLoaderThread(model_dir)
        self.loader_thread.finished_ok.connect(self._on_model_loaded)
        self.loader_thread.failed.connect(self._on_model_failed)
        self.loader_thread.start()

    def _on_model_loaded(self, model, tokenizer, label_map):
        self.model = model
        self.tokenizer = tokenizer
        self.label_map = label_map
        
        self.btn_load_model.setEnabled(True)
        self.btn_predict_manual.setEnabled(True)
        self.btn_predict_selected.setEnabled(self.df is not None)
        
        self._update_status()
        self.statusBar().showMessage("BERT Model structural configurations fully live.")
        
        # Clear out thread instance cleanly
        self.loader_thread.quit()
        self.loader_thread = None

    def _on_model_failed(self, error_msg):
        self.btn_load_model.setEnabled(True)
        QMessageBox.critical(self, "Model Graph Failure", error_msg)
        self.statusBar().showMessage("Initialization sequence failed.")
        self.loader_thread = None

    def _update_status(self):
        model_ok = self.model is not None
        data_ok = self.df is not None
        model_dot = "🟢" if model_ok else "🔴"
        data_dot = "🟢" if data_ok else "🔴"
        self.status_label.setText(
            f"{model_dot} Model Network {'Active' if model_ok else 'Offline'}    "
            f"{data_dot} Dataset Array {'Linked' if data_ok else 'Unlinked'}"
        )

    def on_row_clicked(self, row, _col):
        text_item = self.table.item(row, 1)
        if text_item:
            self.selected_text.setPlainText(text_item.text())

    def predict_selected_row(self):
        text = self.selected_text.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "Invalid Selection", "Please make a selection in the data table first.")
            return
        self._run_prediction(text)

    def predict_manual_text(self):
        text = self.manual_input.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "Empty Sandbox Sequence", "Please pass valid array metrics down to manual block.")
            return
        self.selected_text.setPlainText(text)
        self._run_prediction(text)

    def _run_prediction(self, text):
        if self.model is None or self.tokenizer is None:
            return

        # Interface State Lockout across execution pass
        self.btn_predict_selected.setEnabled(False)
        self.btn_predict_manual.setEnabled(False)
        self.statusBar().showMessage("Executing forward graph propagation pass...")

        self.predict_thread = PredictThread(self.model, self.tokenizer, self.label_map, text)
        self.predict_thread.finished_ok.connect(self._on_prediction_done)
        self.predict_thread.failed.connect(self._on_prediction_failed)
        self.predict_thread.start()

    def _on_prediction_done(self, label, confidence, all_probs):
        self.btn_predict_selected.setEnabled(True)
        self.btn_predict_manual.setEnabled(True)

        # Dynamic Color Matching Scheme Updates
        accent_color = COLOR_MAP.get(label, COLOR_MAP["Primary"])
        
        self.prediction_box.setText(f"{label.upper()} ({confidence:.1f}%)")
        self.prediction_box.setStyleSheet(f"""
            #predictionBox {{
                background-color: {accent_color}1a;
                border: 1px solid {accent_color};
                border-radius: 8px;
                font-size: 18px;
                font-weight: 700;
                color: {accent_color};
            }}
        """)
        
        self.confidence_bar.setValue(int(round(confidence)))
        self.confidence_bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {accent_color}; }}")

        # Dynamically build and paint the Distribution Space Area layout
        self._clear_layout(self.metrics_layout)
        
        for lbl, probability in sorted(all_probs, key=lambda x: x[1], reverse=True):
            lbl_color = COLOR_MAP.get(lbl, "#94a3b8")
            
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 2, 0, 2)
            
            lbl_meta = QLabel(f"{lbl:12}")
            lbl_meta.setFixedWidth(85)
            lbl_meta.setStyleSheet("font-weight: 600; font-size: 12px;")
            
            pbar = QProgressBar()
            pbar.setRange(0, 100)
            pbar.setValue(int(round(probability)))
            pbar.setFormat(f"{probability:.1f}%")
            pbar.setFixedHeight(16)
            pbar.setStyleSheet(f"""
                QProgressBar {{ background-color: #030712; border: 1px solid #1f2937; text-align: right; color: #ffffff; font-size: 10px; }}
                QProgressBar::chunk {{ background-color: {lbl_color}; }}
            """)
            
            row_layout.addWidget(lbl_meta)
            row_layout.addWidget(pbar)
            self.metrics_layout.addWidget(row_widget)

        self.statusBar().showMessage(f"Inference Cycle execution completed successfully.")
        self.predict_thread.quit()
        self.predict_thread = None

    def _on_prediction_failed(self, error_msg):
        self.btn_predict_selected.setEnabled(True)
        self.btn_predict_manual.setEnabled(True)
        QMessageBox.critical(self, "Inference Pass Failure", error_msg)
        self.statusBar().showMessage("Execution pipeline failed.")
        self.predict_thread = None

    def _clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()


# --------------------------------------------------------------------------
def main():
    # Setup High-DPI support constraints
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = SentimentApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()