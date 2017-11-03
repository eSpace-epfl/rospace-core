

class OrbitalState(object):
    def __init__(self):
        self.uptodate = False




class CartesianOrbitalState(OrbitalState):
    def __init__(self):
        super(CartesianOrbitalState, self).__init__()
        self.a = 2


class RelativeOrbitalState(OrbitalState):
    def __init__(self):
        super(RelativeOrbitalState, self).__init__()
        self.a = 3


class OrbitalStateInterface(object):
    def __init__(self, state, facade):
        self.state = state
        self.facade = facade

    def notify_change(self):
        self.facade.notify_change(self)

    def update(self):
        self.facade.update(self)

class CartesianInterface(OrbitalStateInterface):
    state_type = CartesianOrbitalState


    def __init__(self, state, facade):
        super(CartesianInterface, self).__init__(state, facade)
        self.can_convert_form = {KeplerianOrbitalState: self.convert_from_cart}

    @property
    def a(self):
        print "getter a"
        self.update()
        return self.state.a

    @a.setter
    def a(self, value):
        print "setter a"
        self.state.a = value
        self.notify_change()

    def convert_from_cart(self, cartesian):
        print "blab"



class KeplerianInterface(OrbitalStateInterface):
    state_type = KeplerianOrbitalState

    def __init__(self, state, facade):
        super(KeplerianInterface, self).__init__(state, facade)
        self.can_convert_form = {CartesianOrbitalState: self.convert_from_cart}

    @property
    def a(self):
        self.update()
        return self.state.d

    @a.setter
    def a(self, value):
        self.state.d = value
        self.notify_change()

    def convert_from_cart(self, other_state):
        print "blub"





class OrbitalStateFacade:
    def __init__(self):
        self.data = {}
        self.uptodate = set([])

    def get_interface(self, int_type):
        state_type = int_type.state_type

        # if this is the first access
        if not self.data:
            # create state object and set as up to date
            self.data[state_type] = state_type()
            self.uptodate.add(state_type)
            return int_type(self.data[state_type], self)

        # we already have data, but not in this format
        elif state_type not in self.data:
            # create data type and update its data
            self.data[state_type] = state_type()

        # create interface and return
        interface = int_type(self.data[state_type], self)
        self.update(interface)
        return interface

    def update(self, interface):
        state_type = type(interface).state_type
        if state_type in self.uptodate:
            return

        # get possible conversions
        conversion_types = [x for x in interface.can_convert_form if x in self.uptodate]

        if not conversion_types:
            raise("No suitable conversion found!")

        # get pointer to conversion method in interface
        conversion_method = interface.can_convert_form[conversion_types[0]]

        # execute conversion method
        conversion_method(self.data[conversion_types[0]])

        # set as uptodate
        self.uptodate.add(state_type)

    def notify_change(self, interface):
        state_type = type(interface).state_type
        # set only current type as uptodate
        self.uptodate.clear()
        self.uptodate.add(state_type)

    def __str__(self):
        available = "Available: " + str([x.__name__ for x in self.data])
        uptodate = "Up to date: " + str([x.__name__ for x in self.uptodate])
        return available + "\n" + uptodate



if __name__ == '__main__':
    fac = OrbitalStateFacade()

    aint = fac.get_interface(KeplerianInterface)
    print fac

    bint = fac.get_interface(CartesianInterface)
    print fac

    aint.a = 10
    print fac

    bint.a = 5
    print fac

    print aint.a
    print fac



