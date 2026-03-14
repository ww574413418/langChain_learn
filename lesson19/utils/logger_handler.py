import logging
from datetime import datetime
from .path_tool import get_abs_path
import os

# 获取日志文件路径
LOG_ROOT = get_abs_path("logs")
# 确保日志文件存在
os.makedirs(LOG_ROOT, exist_ok=True)

# 日志格式
DEAFAULT_LOG_FORMAT = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)

def get_logger(name: str,
               console_level:int = logging.INFO,#控制台日志级别
               file_level:int = logging.DEBUG,#日志文件日志级别
               log_file = None) -> logging.Logger:
    '''

    :param name: log的名字
    :param console_level:  控制台显示日志的级别
    :param file_level: log文件保存日志的级别
    :param log_file: log文件名字规则
    :return:
    '''
    logger = logging.Logger(name)
    logger.setLevel(logging.DEBUG)

    # 没有会重复打印
    if logger.handlers:
        return logger

    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEAFAULT_LOG_FORMAT)

    logger.addHandler(console_handler)

    #文件handler
    if not log_file:
        log_file = os.path.join(LOG_ROOT,f"{name}_{datetime.now().strftime("%Y%m%d")}.log")

    file_handler = logging.FileHandler(log_file,mode="a",encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEAFAULT_LOG_FORMAT)

    logger.addHandler(file_handler)

    return logger


#快捷获取日志器
logger = get_logger("")

if __name__ == '__main__':
    logger.info("info日志")
    logger.error("错误日志")
    logger.warning("警告日志")
    logger.debug("调试日志") # 不输出 console 日志级别为INFO,log文件里有