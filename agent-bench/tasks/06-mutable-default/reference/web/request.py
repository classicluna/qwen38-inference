class Request:
    def __init__(self, path, headers=None):
        self.path = path
        self.headers = {} if headers is None else headers

    def with_header(self, k, v):
        self.headers[k] = v
        return self
