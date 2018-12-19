# ROSpace

[![pipeline status](https://gitlab.com/eSpace-epfl/rospace/core/badges/master/pipeline.svg)](https://gitlab.com/eSpace-epfl/rospace/core/commits/master)
[![codecov](https://codecov.io/gl/rospace/core/branch/master/graph/badge.svg?token=BKZJjUEI6Y)](https://codecov.io/gl/rospace/core)

### What
ROSpace is a multipurpose simulation tool aimed at simulating physically
 accurate in-orbit satellite operations. It can be used to test among
others rendezvous scenarios, dynamic satellite behaviour, subsystem
interactions, attitude control and HIL (Hardware in the Loop).

It is currently limited to Geocentric orbits but its modular architecture
could potentially allow for other type orbits.

### How
ROSpace is built around the [Orekit toolkit](https://www.orekit.org/)
and ROS [Robot Operating System](http://www.ros.org/). The use of the
Orekit library is especially is important due to having been
[validated](https://www.orekit.org/static/faq.html) in various
real-world scenarios and data.

### Who
ROSpace has been initiated by the
[EPFL Space Engineering Center](https://espace.epfl.ch/) as well as by
members of the [Autonomous Systems Lab](http://www.asl.ethz.ch/) at
ETHZ. All contributors are listed in [AUTHORS.md](AUTHORS.md). Feel free
to submit issues and patches, see [CONTRIBUTING.md](CONTRIBUTING.md) for
details.

### Cut the crap, lets get started
Have a look at the [Wiki](https://gitlab.com/eSpace-epfl/rospace/core/wikis/home)
which has instructions to get started. Even more in a hurry ?
```bash
git clone https://gitlab.com/eSpace-epfl/rospace/spaceport.git
cd spaceport
./build.sh rospace
./run.sh rospace
roslaunch rospace_simulator simulator.launch mission:=envisat_mission scenario:=envisat_mission
```