"""
This module provides custom exception handling functionality, including detailed \
    error message generation.

It contains a function for extracting detailed information from \
    exceptions (file name, line number, and error message) \
and a custom exception class that can be used throughout the application to raise \
    more informative error messages.
"""
import sys

def error_message_detail(error,error_detail:sys):
    """
    Extracts and returns a detailed error message including the file name \
        and line number where the error occurred.

    Args:
        error (Exception): The error or exception instance that was raised.
        error_detail (sys): The `sys` module, used to extract detailed \
            information about the exception.

    Returns:
        str: A formatted error message string containing the script name, \
            line number, and error message.
    """
    _,_,exc_tb=error_detail.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occured in python script name [{filename}] \
        line number [{exc_tb.tb_lineno}] error message [{str(error)}]"
    return error_message

class CustomException(Exception):
    """
    Custom exception class for handling and raising detailed error messages.

    This class extends the built-in `Exception` class and generates a detailed error message \
        using the `error_message_detail` function. It captures the script name, line number, and \
            error message when an exception is raised.

    Attributes:
        error_message (str): A detailed error message generated using the \
            `error_message_detail` function.
    """
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_detail)

    def __str__(self) -> str:
        return self.error_message
