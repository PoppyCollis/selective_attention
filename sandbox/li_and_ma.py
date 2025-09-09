import matplotlib.pyplot as plt
import numpy as np



base = np.eye(3)

matrix_1 = np.flip(base * 10, axis=0)
matrix_1[matrix_1==0] = 0

matrix_2 = np.flip(base * 8, axis=0)
matrix_2[matrix_2==0] = 1

matrix_3 = np.flip(base * 6, axis=0)
matrix_3[matrix_3==0] = 2

matrix_4 = np.flip(base * 4, axis=0)
matrix_4[matrix_4==0] = 3

utility_a_w = np.array([matrix_1,matrix_2, matrix_3, matrix_4]).reshape(-1, 3)


# Display with viridis colormap
plt.imshow(utility_a_w, cmap='viridis')
plt.colorbar()  # adds a color scale on the side
plt.show()

print(utility_a_w)