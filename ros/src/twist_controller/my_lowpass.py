
class MyLowPassFilter(object):
    def __init__(self, tau, Ts):
        self.a = (tau - Ts) / tau
        self.b = Ts / tau

        self.y_s = 0.
        self.ready = False

    def get(self):
        return self.y_s

    def filt(self, u):

        if self.ready:
            self.y_s = self.a * self.y_s + self.b * u
        else:
            self.ready = True

        return self.y_s