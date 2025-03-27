# RUN: python %s 2>&1 | filecheck %s

import xtc.graphs.xtc.op as O

x = O.tensor()

with O.graph(name="relu") as gb:
    O.relu(x)

graph = gb.graph
print(graph)
# CHECK:       graph:
# CHECK-NEXT:    name: relu
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %1
# CHECK-NEXT:    nodes:
# CHECK-NEXT:      %1: relu(%0)
