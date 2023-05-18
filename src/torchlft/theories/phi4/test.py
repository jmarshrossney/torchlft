import torch
from torchlft.theories.phi4.coupling_flow_fnn import *
import torchlft.constraints as constraints

H = [100,]
L = [20, 20]
D = 4

DEVICE = "cuda"

layers_1 = [
    AdditiveCouplingLayer(
        L, hidden_shape=H, activation=torch.nn.Tanh()
    )
    for _ in range(D)
]

layers_2 = [
       AffineCouplingLayer(L, hidden_shape=H, activation=torch.nn.Tanh()) for _ in range(D)
       ]

layers_3 = [
    RQSplineCouplingLayer(
        L,
        n_segments=8,
        upper_bound=5,
        hidden_shape=H,
        activation=torch.nn.Tanh(),
    )
    for _ in range(D)
]

geometry = Geometry(L)

flow = NormalizingFlow(geometry, *layers_3).to(DEVICE).eval()
sample = torch.randn(100, *L, device=DEVICE)

# Test it works
flow(sample)

#torch.jit._state.disable()

print("JIT enabled: ", torch.jit._state._enabled.enabled)

traced = torch.jit.trace(flow, (torch.randn(10, *L, device=DEVICE),))
traced(sample)
#print("Traced ", type(traced))

"""
for name, submodule in traced.named_children():
    try:
        print(name)
        print(submodule.code)
        print("\n\n\n")
    except:
        pass
#print(traced.code)
"""

scripted = torch.jit.script(flow)
scripted(sample)
#print("Scripted:", type(scripted))
#print(scripted.code)

from time import time

N = 1000
R = 10

t1 = time()
for _ in range(R):
    _ = flow(torch.randn(N, *L, device=DEVICE))
t2 = time()
print("No JIT: ", (t2 - t1) / R)
   
t1 = time()
for _ in range(R):
    _ = traced(torch.randn(N, *L, device=DEVICE))
t2 = time()
print("Traced: ", (t2 - t1) / R)
   
t1 = time()
for _ in range(R):
    _ = scripted(torch.randn(N, *L, device=DEVICE))
t2 = time()
print("Scripted: ", (t2 - t1) / R)

