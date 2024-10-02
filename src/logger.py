"""
This module sets up logging functionality for an application.

It creates a unique log file in the 'logs' directory within the current working directory. 
The log file is named using the current timestamp in the format `MM_DD_YYYY_HH_MM_SS.log`, 
ensuring that each execution generates a separate log file.

The logging configuration writes log messages to the generated file in a specific format:
`[timestamp] line_number logger_name - log_level - message`.

Modules and scripts that import and use this module can log messages with various severity levels 
(INFO, WARNING, ERROR, etc.), which will be recorded in the generated log file.

Main components:
- `LOG_FILE`: Filename of the log file based on the current timestamp.
- `log_path`: Directory path where the log file will be stored.
- `LOG_FILE_PATH`: Full path to the log file.
- `logging.basicConfig`: Configures logging with the specified file, format, and log level.
"""

import os
import logging
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

log_path = os.path.join(os.getcwd(),'logs',LOG_FILE)
os.makedirs(log_path,exist_ok=True)

LOG_FILE_PATH = os.path.join(log_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format= "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO)
