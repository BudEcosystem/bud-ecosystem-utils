import logging
# from elasticapm.handlers.logging import LoggingHandler
# from elasticapm.contrib.starlette import make_apm_client
# from pythonjsonlogger import jsonlogger
import os


# Setup Elastic APM configuration
apm_config = {
    "SERVICE_NAME":  os.getenv("APM_SERVICE_NAME", "my-fastapi-service"),
    "SECRET_TOKEN": os.getenv("APM_SECRET_TOKEN", ""),
    "SERVER_URL": os.getenv("APM_SERVER_URL", "http://localhost:8200"),
    "ENVIRONMENT": os.getenv("APM_ENVIRONMENT", "development"),
}

# apm_client = make_apm_client(apm_config)  # Create an APM client instance


def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
    """
    Configure and return a logger instance.

    :param name: Name of the logger.
    :param level: Logging level. Defaults to logging.INFO.
    :return: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Setup Elastic APM Logging Handler
    # apm_handler = LoggingHandler(client=apm_client)
    # apm_handler.setLevel(logging.ERROR)  # Set the logging level for APM logs

    # Setup Console Logging Handler
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(level)

    # # Create a formatter and set it for both handlers
    # formatter = logging.Formatter(
    #     "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    # )

    # formatterConsole = jsonlogger.JsonFormatter(
    #     fmt="%(asctime)s %(levelname)s %(name)s %(message)s %(pathname)s %(filename)s %(lineno)d",
    #     datefmt="%Y-%m-%dT%H:%M:%S",
    # )

    # # Setup Logstach Format

    # #apm_handler.setFormatter(formatterConsole)
    # console_handler.setFormatter(formatterConsole)

    # # Add Handlers to the Logger
    # logger.addHandler(console_handler)
    # #logger.addHandler(apm_handler)

    return logger