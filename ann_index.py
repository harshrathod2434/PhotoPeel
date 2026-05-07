import numpy as np


class LSHIndex:
    def __init__(self, dim, num_tables=6, num_bits=16, seed=42):
        self.dim = int(dim)
        self.num_tables = int(num_tables)
        self.num_bits = int(num_bits)
        rng = np.random.default_rng(seed)
        self.planes = [rng.standard_normal((self.num_bits, self.dim), dtype=np.float32)
                       for _ in range(self.num_tables)]
        self.tables = [dict() for _ in range(self.num_tables)]
        self.vectors = {}
        self.hashes = {}
        self._next_id = 0

    def _hash_vec(self, vec, planes):
        proj = planes @ vec
        bits = proj >= 0
        h = 0
        for b in bits:
            h = (h << 1) | int(b)
        return h

    def add(self, vector, item_id=None):
        vec = np.asarray(vector, dtype=np.float32)
        if item_id is None:
            item_id = self._next_id
            self._next_id += 1
        hashes = []
        for t in range(self.num_tables):
            h = self._hash_vec(vec, self.planes[t])
            hashes.append(h)
            bucket = self.tables[t].setdefault(h, [])
            bucket.append(item_id)
        self.vectors[item_id] = vec
        self.hashes[item_id] = hashes
        return item_id

    def update(self, item_id, vector):
        vec = np.asarray(vector, dtype=np.float32)
        old_hashes = self.hashes.get(item_id)
        if old_hashes is None:
            return self.add(vec, item_id=item_id)
        for t, h in enumerate(old_hashes):
            bucket = self.tables[t].get(h)
            if bucket:
                try:
                    bucket.remove(item_id)
                except ValueError:
                    pass
                if not bucket:
                    del self.tables[t][h]
        hashes = []
        for t in range(self.num_tables):
            h = self._hash_vec(vec, self.planes[t])
            hashes.append(h)
            bucket = self.tables[t].setdefault(h, [])
            bucket.append(item_id)
        self.vectors[item_id] = vec
        self.hashes[item_id] = hashes
        return item_id

    def query(self, vector, max_candidates=None):
        vec = np.asarray(vector, dtype=np.float32)
        candidates = set()
        for t in range(self.num_tables):
            h = self._hash_vec(vec, self.planes[t])
            bucket = self.tables[t].get(h)
            if bucket:
                candidates.update(bucket)
        if not candidates:
            return []
        if max_candidates is not None and len(candidates) > max_candidates:
            return sorted(candidates)[:max_candidates]
        return list(candidates)
