import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# 数据库连接配置
# 推荐格式: mysql+pymysql://<username>:<password>@<host>:<port>/<dbname>?charset=utf8mb4
# TODO: 根据你的本地 MySQL 服务调整为正确的密码和数据库名，例如新建名为 insulator_db 的库
DATABASE_URL = "mysql+pymysql://root:10.26syd@localhost:3306/insulator_db?charset=utf8mb4"

Base = declarative_base()

class InspectionRecord(Base):
    """巡检记录表"""
    __tablename__ = 'inspection_records'

    id = Column(Integer, primary_key=True, autoincrement=True, comment="记录ID")
    inspector_name = Column(String(50), nullable=True, default="Anonymous", comment="巡检员姓名")
    inspection_time = Column(DateTime, default=datetime.datetime.now, comment="检测时间")
    source_type = Column(String(20), nullable=False, comment="检测源类型: image / video")
    file_path = Column(String(255), nullable=False, comment="本地文件路径")
    status = Column(String(20), default="done", comment="状态: processing / done")
    
    # 关联到识别结果列表
    results = relationship("DetectionResult", back_populates="record", cascade="all, delete-orphan")


class DetectionResult(Base):
    """具体检测结果表(单张图/单帧视频中的异常框信息)"""
    __tablename__ = 'detection_results'

    id = Column(Integer, primary_key=True, autoincrement=True, comment="结果ID")
    record_id = Column(Integer, ForeignKey('inspection_records.id'), nullable=False, comment="关联的巡检记录ID")
    class_name = Column(String(50), nullable=False, comment="类别名称(如: broken_insulator)")
    confidence = Column(Float, nullable=False, comment="置信度")
    bbox = Column(String(100), nullable=True, comment="边界框 [x1, y1, x2, y2]")
    
    # 关联回巡检记录
    record = relationship("InspectionRecord", back_populates="results")


class ModelMetric(Base):
    """模型性能指标对比表"""
    __tablename__ = 'model_metrics'

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False, comment="模型名称")
    mAP50 = Column(Float, nullable=True, comment="mAP@0.5")
    fps = Column(Float, nullable=True, comment="推理速度 FPS")
    params_m = Column(Float, nullable=True, comment="参数量(M)")
    notes = Column(Text, nullable=True, comment="备注(如使用了哪些改进模块，如 WTFConv, Mamba 等)")


def init_db():
    """初始化数据库并同步创建所有数据表"""
    engine = create_engine(DATABASE_URL, echo=False)
    # 创建所有继承自 Base 的表，若表已存在则不会操作
    Base.metadata.create_all(engine)
    return engine

def get_session():
    """获取一个数据库连接会话用于 CRUD 操作"""
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    return Session()
