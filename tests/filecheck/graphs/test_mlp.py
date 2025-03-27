# RUN: python %s 2>&1 | filecheck %s

import xtc.graphs.xtc.op as O

img_size = 32*32*3
img = O.tensor(shape=(img_size,))
w1 = O.tensor(shape=(img_size, 512))
w2 = O.tensor(shape=(512, 256))
w3 = O.tensor(shape=(256, 128))
w4 = O.tensor(shape=(128, 10))

# Mulit Layer Perceptron with 3 relu(fc) + 1 fc
with O.graph(name="mlp4") as gb:
    with O.graph(name="l1"):
        l1 = O.matmul(img, w1)
        l1 = O.relu(l1)
    with O.graph(name="l2"):
        l2 = O.matmul(l1, w2)
        l2 = O.relu(l2)
    with O.graph(name="l3"):
        l3 = O.matmul(l2, w3)
        l3 = O.relu(l3)
    with O.graph(name="l4"):
        l4 = O.relu(O.matmul(l3, w4))

mlp4 = gb.graph
print(mlp4)
# CHECK:       graph:
# CHECK-NEXT:    name: mlp4
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0
# CHECK-NEXT:    - %1
# CHECK-NEXT:    - %2
# CHECK-NEXT:    - %3
# CHECK-NEXT:    - %4
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %12
# CHECK-NEXT:    nodes:
# CHECK-NEXT:      %5: matmul(%0, %1)
# CHECK-NEXT:      %6: relu(%5)
# CHECK-NEXT:      %7: matmul(%6, %2)
# CHECK-NEXT:      %8: relu(%7)
# CHECK-NEXT:      %9: matmul(%8, %3)
# CHECK-NEXT:      %10: relu(%9)
# CHECK-NEXT:      %11: matmul(%10, %4)
# CHECK-NEXT:      %12: relu(%11)
