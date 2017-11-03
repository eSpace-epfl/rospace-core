from OrbitalStateFacade import *


class KepOrbState(OrbitalState):

    def __init__(self):
        super(KepOrbState, self).__init__()
        self.a = 0
        self.e = 0
        self.O = 0
        self.w = 0
        self.i = 0


class MeanAnKepOrbState(KepOrbState):
    def __init__(self):
        super(MeanAnKepOrbState, self).__init__()
        self.m = 0


class MeanMotionKepOrbState(KepOrbState):
    def __init__(self):
        super(MeanMotionKepOrbState, self).__init__()
        self.n = 0


class EccKepOrbState(KepOrbState):
    def __init__(self):
        super(EccKepOrbState, self).__init__()
        self.E = 0


class TrueKepOrbState(KepOrbState):
    def __init__(self):
        super(TrueKepOrbState, self).__init__()
        self.v = 0

# interfaces



class KepOrbIface(OrbitalStateInterface):
    state_type = None  # abstract

    def __init__(self, state, facade):
        super(KepOrbIface, self).__init__(state, facade)
        self.can_convert_form = {} # abstract

    # getter / setter
    @property
    def a(self):
        self.update()
        return self.state.a

    @a.setter
    def a(self, value):
        self.state.a = value
        self.notify_change()

    @property
    def e(self):
        self.update()
        return self.state.e

    @e.setter
    def e(self, value):
        self.state.e = value
        self.notify_change()

    @property
    def O(self):
        self.update()
        return self.state.O

    @O.setter
    def O(self, value):
        self.state.O = value
        self.notify_change()

    @property
    def w(self):
        self.update()
        return self.state.w

    @w.setter
    def w(self, value):
        self.state.w = value
        self.notify_change()

    @property
    def i(self):
        self.update()
        return self.state.i

    @i.setter
    def i(self, value):
        self.state.i = value
        self.notify_change()


class MeanMotionKepOrbIface(KepOrbIface):
    state_type = MeanMotionKepOrbState  # abstract

    def __init__(self, state, facade):
        super(MeanMotionKepOrbIface, self).__init__(state, facade)
        self.can_convert_form = {TrueKepOrbState: conv_from_truekep}  # abstract

    @property
    def m(self):
        self.update()
        return self.state.m

    @m.setter
    def m(self, value):
        self.state.m = value
        self.notify_change()


    def conv_from_truekep(self, other):
        self.a = other.a
        self.e = other.e
        self.w = other.w
        self.O = other.O
        self.i = other.i
