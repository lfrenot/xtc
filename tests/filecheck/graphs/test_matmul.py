# RUN: python %s 2>&1 | filecheck %s

import xtc.graphs.xtc.op as O

x = O.tensor()
y = O.tensor()

with O.graph(name="matmul") as gb:
    O.matmul(x, y)

graph = gb.graph
print(graph)
# CHECK:       graph:
# CHECK-NEXT:    name: matmul
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0
# CHECK-NEXT:    - %1
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %2
# CHECK-NEXT:    nodes:
# CHECK-NEXT:      %2: matmul(%0, %1)
