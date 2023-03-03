class Buffer():
    def __init__(self) -> None:
        self.images = []
        self.ids = []
    def add(self, step, image):
        self.ids.append(step)
        self.images.append(image)
    def get(self, index):
        if index > len(self.images):
            print(f"Index {index} out of range {len(self.images)}")
        return self.images[index]