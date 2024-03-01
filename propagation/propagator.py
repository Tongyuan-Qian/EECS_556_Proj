import cv2
import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt


class Propagator:
    def __init__(self, mode: str = "default", mono_bgr=None, hint_bgr=None, mono_file=None, hint_file=None):
        if mode == "default" and mono_bgr is not None and hint_bgr is not None:
            self.mono_bgr = mono_bgr
            self.hint_bgr = hint_bgr
        elif mode == "read file" and mono_file is not None and hint_file is not None:
            self.mono_bgr = cv2.imread(mono_file)
            self.hint_bgr = cv2.imread(hint_file)
        else:
            raise ValueError("Invalid Arguments")

        self.mono_yuv = cv2.cvtColor(self.mono_bgr, cv2.COLOR_BGR2YUV) / 255.0
        self.hint_yuv = cv2.cvtColor(self.hint_bgr, cv2.COLOR_BGR2YUV) / 255.0

    def get_init_weights(self, x, y, n, m, d=1):
        idx = np.array([self.pos_to_idx(x, y, m)])
        Y = np.array([self.mono_yuv[x, y, 0]])
        for i in range(max(0, x - d), min(n, x + d + 1)):
            for j in range(max(0, y - d), min(m, y + d + 1)):
                if i != x or j != y:
                    idx = np.append(idx, self.pos_to_idx(i, j, m))
                    Y = np.append(Y, self.mono_yuv[i, j, 0])

        # choose one:
        # sigma = np.std(Y)   # with center pixel
        sigma = np.std(Y[1:]) # without center pixel

        w = np.array([])
        for i in range(len(idx)):
            if sigma > 1e-3:
                w = np.append(w, np.exp(-((Y[i] - Y[0]) ** 2) / (2 * sigma ** 2)))
            else:
                w = np.append(w, 1.)

        w[1:] = -w[1:] / (np.sum(w) - 1)
        return idx, w

    def propagate(self):
        n, m, _ = self.hint_yuv.shape
        size = n * m
        W = sparse.lil_matrix((size, size), dtype=float)
        u_hint = np.zeros(size)
        v_hint = np.zeros(size)
        for i in range(n):
            for j in range(m):
                if np.sum(np.abs(self.mono_yuv[i, j] - self.hint_yuv[i, j])) > 1e-2:
                    idx = self.pos_to_idx(i, j, m)
                    W[idx, idx] = 1.
                    u_hint[idx] = self.hint_yuv[i, j, 1]
                    v_hint[idx] = self.hint_yuv[i, j, 2]
                    continue

                idx, w = self.get_init_weights(i, j, n, m)
                for k in range(len(idx)):
                    W[idx[0], idx[k]] = w[k]

        W = W.tocsc()
        u = sparse.linalg.spsolve(W, u_hint)
        v = sparse.linalg.spsolve(W, v_hint)

        res = np.zeros((n, m, 3))
        res[:, :, 0] = self.mono_yuv[:, :, 0]
        for i in range(n):
            for j in range(m):
                idx = self.pos_to_idx(i, j, m)
                res[i, j, 1] = u[idx]
                res[i, j, 2] = v[idx]

        res_yuv = (np.clip(res, 0., 1.) * 255).astype(np.uint8)
        res_rgb = cv2.cvtColor(res_yuv, cv2.COLOR_YUV2RGB)
        res_bgr = cv2.cvtColor(res_yuv, cv2.COLOR_YUV2BGR)
        return res_yuv, res_rgb, res_bgr

    @staticmethod
    def pos_to_idx(x, y, m):
        return x * m + y


def plot_results(res_rgb):
    plt.imshow(res_rgb)
    plt.axis('off')
    plt.show()


# a simple test:
prop = Propagator(mode="read file", mono_file="mono.png", hint_file="hint.png")
res = prop.propagate()
plot_results(res[1])


