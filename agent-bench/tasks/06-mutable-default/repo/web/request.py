class Request:
    def __init__(self, path, headers={}):
        self.path = path
        self.headers = headers

    def with_header(self, k, v):
        self.headers[k] = v
        return self
