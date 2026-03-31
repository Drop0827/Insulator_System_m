import sys
from PyQt5.QtWidgets import QApplication
from db.models import init_db
from gui.main_window import MainWindow

def main():
    # 1. 初始化数据库表结构
    try:
        init_db()
        print("[System Success] MySQL Database Ready.")
    except Exception as e:
        print(f"[Database Error] Check MySQL status: {e}")
        
    app = QApplication(sys.argv)
    
    # 2. 应用白色明亮主题样式 (Modern Light Theme)
    # 取消之前的深色背景，改为浅灰色与白色为主
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f7f7f9;
        }
        QWidget {
            font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
        }
        QLabel {
            color: #444;
            font-size: 14px;
        }
        QPushButton {
            background-color: #ffffff;
            color: #333;
            border: 1px solid #ccd0d5;
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: 500;
        }
        QPushButton:hover {
            background-color: #f2f2f2;
            border-color: #0078D4;
            color: #0078D4;
        }
        QPushButton:pressed {
            background-color: #e5e5e5;
        }
        QListWidget {
            background-color: #ffffff;
            color: #2c3e50;
            border: 1px solid #dcdde1;
            padding: 5px;
            border-radius: 4px;
            font-family: Consolas, 'Courier New', monospace;
            font-size: 13px;
        }
        QGroupBox {
            color: #2f3640;
            font-weight: bold;
            font-size: 15px;
            border: 1px solid #dcdde1;
            border-radius: 8px;
            margin-top: 20px;
            padding-top: 15px;
            background-color: #ffffff;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 10px;
            color: #0078D4;
        }
        QTableWidget {
            background-color: #ffffff;
            alternate-background-color: #f8f9fa;
            gridline-color: #dee2e6;
            selection-background-color: #0078D4;
            color: #333;
        }
        QHeaderView::section {
            background-color: #f1f2f6;
            padding: 4px;
            border: 1px solid #dcdde1;
            font-weight: bold;
        }
    """)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
