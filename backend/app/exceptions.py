from __future__ import annotations


class ApplicationError(Exception):
    """
    Base exception for expected application-level errors.

    These exceptions represent failures that are safe to expose
    to API clients in a controlled form.
    """

    status_code: int = 400
    error_code: str = "APPLICATION_ERROR"
    public_message: str = "The request could not be processed."

    def __init__(
        self,
        message: str | None = None,
        *,
        status_code: int | None = None,
        error_code: str | None = None,
    ) -> None:
        super().__init__(message or self.public_message)

        if status_code is not None:
            self.status_code = status_code

        if error_code is not None:
            self.error_code = error_code

        self.message = message or self.public_message


class ValidationApplicationError(ApplicationError):
    """
    Raised when application-level validation fails.
    """

    status_code = 400
    error_code = "VALIDATION_ERROR"
    public_message = "The supplied data is invalid."


class ResourceNotFoundError(ApplicationError):
    """
    Raised when a required application resource cannot be found.
    """

    status_code = 404
    error_code = "RESOURCE_NOT_FOUND"
    public_message = "The requested resource was not found."


class PredictionApplicationError(ApplicationError):
    """
    Raised when a prediction cannot be completed because of an
    expected application/model processing problem.
    """

    status_code = 400
    error_code = "PREDICTION_ERROR"
    public_message = "The prediction could not be completed."


class ConfigurationApplicationError(ApplicationError):
    """
    Raised when application configuration is invalid.
    """

    status_code = 500
    error_code = "CONFIGURATION_ERROR"
    public_message = "The application configuration is invalid."