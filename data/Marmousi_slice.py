import torch
import cv2
import matplotlib.pylab as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ny = 2301
nx = 751
dx = 4.0

v = torch.from_file('.\marmousi_vp.bin', size=ny*nx).reshape(ny, nx).to(device)
v = v.cpu().T.numpy()

v = cv2.resize(v, (1200, 380), interpolation=cv2.INTER_LINEAR)

plt.imshow(v[:, :])
plt.show()

v = np.clip(v, 1500, 4500)


ind = 0
save_vmodel = np.zeros([20, 1, 128, 128]).astype(np.float32)

len = 1200 // 200
for i in range(len):
    resized = cv2.resize(v[0: 200, i * 200: (i + 1) * 200], (128, 128), interpolation=cv2.INTER_LINEAR)
    save_vmodel[ind][0] = resized
    ind += 1
    plt.imshow(resized, vmax=4500, vmin=1500)
    plt.show()

len = 1200 // 245
for i in range(len):
    resized = cv2.resize(v[0: 245, i * 245: (i + 1) * 245], (128, 128), interpolation=cv2.INTER_LINEAR)
    save_vmodel[ind][0] = resized
    ind += 1
    plt.imshow(resized, vmax=4500, vmin=1500)
    plt.show()

len = 1200 // 290
for i in range(len):
    resized = cv2.resize(v[0: 290, i * 290: (i + 1) * 290], (128, 128), interpolation=cv2.INTER_LINEAR)
    save_vmodel[ind][0] = resized
    ind += 1
    plt.imshow(resized, vmax=4500, vmin=1500)
    plt.show()

len = 1200 // 335
for i in range(len):
    resized = cv2.resize(v[0: 335, i * 335: (i + 1) * 335], (128, 128), interpolation=cv2.INTER_LINEAR)
    save_vmodel[ind][0] = resized
    ind += 1
    plt.imshow(resized, vmax=4500, vmin=1500)
    plt.show()

len = 1200 // 380
for i in range(len):
    resized = cv2.resize(v[0: 380, i * 380: (i + 1) * 380], (128, 128), interpolation=cv2.INTER_LINEAR)
    save_vmodel[ind][0] = resized
    ind += 1
    plt.imshow(resized, vmax=4500, vmin=1500)
    plt.show()

# np.save(r".\mvmodel20.npy", save_vmodel)