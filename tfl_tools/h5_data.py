import h5py

class h5gen:
    def __init__(self, file):
        self.file = file

    def __call__(self):
        with h5py.File(self.file, 'r') as hf:
            for im, lbl in hf["X"], hf["Y"]:
                yield im, lbl
