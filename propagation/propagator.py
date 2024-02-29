import cv2


class Propagator:
    def __init__(self, mode: str = "default", mono_bgr=None, hint_bgr=None, mono_file=None, hint_file=None):
        if mode == "default" and mono_bgr is not None and hint_bgr is not None:
            self.ori_mono_bgr = mono_bgr
            self.ori_hint_bgr = hint_bgr
        elif mode == "read file" and mono_file is not None and hint_file is not None:
            self.ori_mono_bgr = cv2.imread(mono_file)
            self.ori_hint_bgr = cv2.imread(hint_file)
        else:
            raise ValueError("Invalid Arguments")

        self.ori_mono_yuv = cv2.cvtColor(self.ori_mono_bgr, cv2.COLOR_BGR2YUV)
        self.ori_hint_yuv = cv2.cvtColor(self.ori_hint_bgr, cv2.COLOR_BGR2YUV)

    def propagate(self):

