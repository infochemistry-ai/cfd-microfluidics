"""Safe errors raised while resolving local CFD stage inputs."""


class StageInputError(RuntimeError):
    def __init__(self, message: str, *, code: str, http_status: int) -> None:
        super().__init__(message)
        self.code = code
        self.http_status = http_status
