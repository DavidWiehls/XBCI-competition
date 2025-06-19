import sys
from typing import Optional
from loguru import logger


class SDKLogger:

    def __init__(self, level: str = "WARNING"):
        self._sdk_logger = None
        self._handler_id = None
        self._initialized = False
        self.format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | <level>{level: <8}</level> | SDK | {name}:{function}:{line} | {message}"
        
        self.set_level(level)

    def _setup_logger(self, level: str):
        """初始化 SDK 日志记录器"""
        if self._initialized:
            print(f'logger already initialized')
            return

        # 创建绑定上下文的 SDK logger
        self._sdk_logger = logger.bind(sdk_internal=True)

        # 添加 handler
        self._handler_id = logger.add(
            sys.stderr,
            level=level,
            format=self.format,
            filter=lambda record: record["extra"].get("sdk_internal", False),
        )
        self._initialized = True

    def set_level(self, level: str):
        """动态设置日志级别"""
        if self._handler_id is not None:
            try:
                logger.remove(self._handler_id)
            except (ValueError, KeyError, AttributeError):
                pass
            self._handler_id = None

        # 重新添加 handler 并保存 handler_id
        self._handler_id = logger.add(
            sys.stderr,
            level=level,
            format=self.format,
            filter=lambda record: record["extra"].get("sdk_internal", False),
        )

    def disable(self):
        """禁用 SDK 日志输出"""
        if self._handler_id is not None:
            try:
                logger.remove(self._handler_id)
            except ValueError:
                pass
            self._handler_id = None

    def enable(self, level: str = "INFO"):
        """重新启用 SDK 日志输出"""
        if self._handler_id is None:
            self._handler_id = logger.add(
                sys.stderr,
                level=level,
                format=self.format,
                filter=lambda record: record["extra"].get("sdk_internal", False),
            )

    @property
    def logger(self):
        """获取 SDK 日志记录器"""
        if not self._initialized:
            self._setup_logger(level="WARNING")
        return self._sdk_logger


sdk_logger_manager = SDKLogger()
sdk_logger = sdk_logger_manager.logger
