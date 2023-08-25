class UnexpectedGPTResponse(Exception):
    """
    Raised when the GPT response does not conform to the expected json schema

    """


class InvalidFunctionResponse(Exception):
    """
    GPT passed an invalid function name back to the caller

    """

    def __init__(self, invalid_function_name: str):
        super().__init__(f"Invalid function name: {invalid_function_name}")
        self.invalid_function_name = invalid_function_name


class InvalidFunctionParameters(Exception):
    """
    GPT passed invalid function parameters back to the caller

    """

    def __init__(self, invalid_function_name: str, invalid_parameters: str):
        super().__init__(f"Invalid function parameters: {invalid_parameters}")
        self.invalid_function_name = invalid_function_name
        self.invalid_parameters = invalid_parameters
