def get_bbox_center(bbox):
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2


class Person:
    def __init__(self, bbox, embedding, similarity=-1):
        self.bbox = bbox
        self.pos = get_bbox_center(bbox)
        self.embedding = embedding
        self.similarity = similarity

    def update(self, bbox, embedding, similarity=-1):
        self.bbox = bbox
        self.pos = get_bbox_center(bbox)
        self.embedding = embedding
        self.similarity = similarity
    
    def __lt__(self, other):
        return (self.bbox[0] + self.bbox[2]) < (other.bbox[0] + other.bbox[2])
