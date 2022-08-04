class Key:
    def __init__(self, inId, outId):
        self.in_idx = inId
        self.out_idx = outId

    def __eq__(self, other):
        return self.in_idx == other.in_idx and self.out_idx == other.out_idx

    def __hash__(self):
        return hash(f'{self.in_idx}{self.out_idx}')

    def __repr__(self):
        return f'{self.in_idx}-{self.out_idx}'
