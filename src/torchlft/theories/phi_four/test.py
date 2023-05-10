import torch


class TensorField:
    def __init__(self, data, **kwargs):
        self._tensor = torch.as_tensor(data, **kwargs)
        self._lattice = data.shape

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        lattices = tuple(a._lattice for a in args if hasattr(a, "_lattice"))
        args = [a._tensor if hasattr(a, "_tensor") else a for a in args]
        assert len(set(lattices)) == 1
        ret = func(*args, **kwargs)
        return cls(ret)

    @property
    def tensor(self):
        return self._tensor

    def __add__(self, other: "TensorField"):
        assert other._lattice == self._lattice
        return type(self)(self._tensor + other._tensor)

    def __sub__(self, other: "TensorField"):
        assert other._lattice == self._lattice
        return type(self)(self._tensor - other._tensor)


t1 = TensorField(torch.rand(10, 10))
t2 = TensorField(torch.rand(10, 10))
t3 = TensorField(torch.rand(10, 1))
t4 = torch.rand(10, 10)

print((t1 + t2)._lattice)
# print((t1 + t2 - torch.add(t1, t2))._tensor)

try:
    torch.add(t1, t3)
except AssertionError:
    pass

print(torch.add(t1, t4)._lattice)

# t1.tensor = torch.add(t1.tensor, t4)

torch.tanh(t1)
