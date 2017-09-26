class BaseRelativeOrbitalFilter:

    def callback_aon(self, angles):
        print "aon-callback"

    def callback_state(self, oe):
        print "state-callback"

    def get_state(self):
        return ["hallo"]

