from . import modelobjects
from . import dataobjects
from . import plotobjects

__doc__ = f"""
In CHUMP’s object-oriented framework, objects serve as the foundational components.
Each object can be defined by a dictionary structure. To declare and create objects,
there are three primary categories of top-level classes: data, model and visualization
classes, each with distinct objectives. The CHUMP framework is designed to be modular
and extensible, allowing new types of classes to be easily added to the system. The major
categories of classes and objects are discussed as follow:

1.	`.dataobjects`: {dataobjects.__doc__}

2. `.modelobjects`: {modelobjects.__doc__}

3.	`.plotobjects`: {plotobjects.__doc__}

"""
