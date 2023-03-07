class Buffer():
    def __init__(self) -> None:
        self.images = {}
    def add(self, step, image):
        self.images[step] = image
    def get(self, index):
        return self.images[index]