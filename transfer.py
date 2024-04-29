import numpy as np
import copy 

class transfer(object):
    # 还是整一个这个啊，简单点，别整乱了
    def __init__(self, **kargs):
        self.higher_state_real = np.array(kargs["higher_state_real"])
        self.higher_state_norm = np.array(kargs["higher_state_norm"])
        self.lower_state_real = np.array(kargs["lower_state_real"])
        self.lower_state_norm = np.array(kargs["lower_state_norm"])
        self.dim = len(self.higher_state_real)

        self.__init_bili()

    def __init_bili(self):
        # 这样每个transfer就只用算一次了，鉴定为好
        self.real_center = 1/2 * (self.higher_state_real + self.lower_state_real)
        self.norm_center = 1/2 * (self.higher_state_norm + self.lower_state_norm)

        self.real_range = self.higher_state_real - self.lower_state_real
        self.norm_range = self.higher_state_norm - self.lower_state_norm
        self.bili_norm_to_real = np.divide(self.real_range, self.norm_range)
        self.bili_real_to_norm = np.divide(self.norm_range, self.real_range)

    def real_to_norm(self, state_real):
        # 就转换呗
        state_real = copy.deepcopy(state_real)
        state_real = self.clip_real(state_real)

        # 然后开始算了

        state_diff_real = state_real - self.real_center
        state_diff_norm = np.multiply(state_diff_real, self.bili_real_to_norm)

        state_norm = state_diff_norm + self.norm_center
        return state_norm

    def norm_to_real(self, state_norm):
        # 也是就转换呗
        state_norm = copy.deepcopy(state_norm)
        state_norm = self.clip_norm(state_norm)

        state_diff_norm = state_norm - self.norm_center
        state_diff_real = np.multiply(state_diff_norm,self.bili_norm_to_real)

        state_real = state_diff_real + self.real_center
        return state_real

    def clip_real(self, state_real):
        flag_clipped = False

        for i in range(self.dim):
            if state_real[i] > self.higher_state_real[i]:
                state_real[i] = self.higher_state_real[i]
            elif state_real[i] < self.lower_state_real[i]:
                state_real[i] = self.lower_state_real[i]
            else:
                flag_clipped = True

        if flag_clipped:
            print("transfer: attenstion, state_real clipped.")

        return  state_real

    def clip_norm(self, state_norm):
        flag_clipped = False
        for i in range(self.dim):
            if state_norm[i] > self.higher_state_norm[i]:
                state_norm[i] = self.higher_state_norm[i]
            elif state_norm[i] < self.lower_state_norm[i]:
                state_norm[i] = self.lower_state_norm[i]
            else:
                flag_clipped = True
        if flag_clipped:
            print("transfer: attenstion, state_norm clipped.")

        return  state_norm
