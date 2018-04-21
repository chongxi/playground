import numpy as np
import collections

#------------------------------------------------------------------------------
# Compare the content of two list/array in a orderless manner
#------------------------------------------------------------------------------
compare = lambda x, y: collections.Counter(x) == collections.Counter(y)

#------------------------------------------------------------------------------
# Convert rgb to gray
#------------------------------------------------------------------------------
def rgb2gray(rgb_color, absolute_value=True):
    """rgt_color is a n by 3 matrix. three column array [r; g; b]
    """
    gray_transform_mat = np.repeat(np.array([0.5989, 0.6870, 0.4140]), 3).reshape(-1,3)
    gray_color = rgb_color.dot(gray_transform_mat)
    if absolute_value:
        return abs(gray_color)
    else:
        return gray_color


#------------------------------------------------------------------------------
# get the normal vector from a camera
#------------------------------------------------------------------------------
def normal_vector(cam):
    """cam contains azimuth and elevation, we use that to get the normal_vector
    of the rendered view plane. This is the basis of ray tracing
    """
    alpha, belta = np.array([cam.azimuth, cam.elevation]) * np.pi / 180
    a = np.sin(alpha) * np.cos(belta)
    b = np.cos(alpha) * np.cos(belta)
    c = np.sin(belta)
    return np.array([a,b,c])


#------------------------------------------------------------------------------
# Generate a line array in 3D space from a Vector and a Point
#------------------------------------------------------------------------------
def generate_line(u0, A):
    '''
    line function:
    u = u0 + At
    '''
    u0 = np.array(u0).astype(np.float32)
    A  = np.array(A).astype(np.float32)
    t = np.linspace(-10000,10000,1000)
    u0 = u0.reshape(-1,1)
    A  = A.reshape(-1,1)
    u = u0 + A*t
    u = u.T
    return u

#------------------------------------------------------------------------------
# Project a vector to xy plane (z=0)
#------------------------------------------------------------------------------
def proj_to_xyplane(p0, A):
    '''
    xy plane is z=0
    p1 is a point (x,y,0) in plane xy
    p1 + At = p0
    '''
    (x0,y0,z0) = p0
    (a, b, c ) = A
    t = -z0/c
    p1 = p0 + A*t
    return p1


#------------------------------------------------------------------------------
# Line Plane Intersection
#------------------------------------------------------------------------------
def line_plane_intersection(rayDirection, rayPoint, planeNormal, planePoint, epsilon=1e-6):
    """The camera view is like ray. The view is the plane which has normal vector as ray direction
    This function calculate the intersecion coordination of a line (defined by rayDirection and rayPoint) 
    and a plane (defined by planeNormal and planePoint)
    https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
    """
    rayDirection = np.array(rayDirection)
    rayPoint = np.array(rayPoint)
    planeNormal = np.array(planeNormal)
    planePoint = np.array(planePoint)
    angle = np.dot(planeNormal, rayDirection) 
    if abs(angle) < epsilon:
        print("no intersection")
    v = rayPoint - planePoint
    s = -np.dot(planeNormal, v) / angle
    point = s * rayDirection + v + planePoint
    return point


#------------------------------------------------------------------------------
# 2d mouse event to 3d coordinate
#------------------------------------------------------------------------------
def pos2d_to_pos3d(pos, cam):
    """Convert mouse event pos:(x, y) into x, y, z translations"""
    """dist is the distance between (x,y) and (cx, cy) of cam"""
    center = get_center_of_view(cam)
    dist = pos - center
    dist[1] *= -1
    rae = np.array([cam.azimuth, cam.elevation]) * np.pi / 180
    saz, sel = np.sin(rae)
    caz, cel = np.cos(rae)
    dx = (+ dist[0] * (1 * caz)
          + dist[1] * (- 1 * sel * saz))
    dy = (+ dist[0] * (1 * saz)
          + dist[1] * (+ 1 * sel * caz))
    dz = (+ dist[1] * 1 * cel)

    # Black magic part 2: take up-vector and flipping into account
    ff = cam._flip_factors
    up, forward, right = cam._get_dim_vectors()
    dx, dy, dz = right * dx + forward * dy + up * dz
    dx, dy, dz = ff[0] * dx, ff[1] * dy, ff[2] * dz
    return dx, dy, dz


#------------------------------------------------------------------------------
# Event system
#------------------------------------------------------------------------------
class EventEmitter(object):
    """Class that emits events and accepts registered callbacks.
    Derive from this class to emit events and let other classes know
    of occurrences of actions and events.
    Example
    -------
    ```python
    class MyClass(EventEmitter):
        def f(self):
            self.emit('my_event', 1, key=2)
    o = MyClass()
    # The following function will be called when `o.f()` is called.
    @o.connect
    def on_my_event(arg, key=None):
        print(arg, key)
    ```
    """

    def __init__(self):
        self._reset()

    def _reset(self):
        """Remove all registered callbacks."""
        self._callbacks = defaultdict(list)

    def _get_on_name(self, func):
        """Return `eventname` when the function name is `on_<eventname>()`."""
        r = re.match("^on_(.+)$", func.__name__)
        if r:
            event = r.group(1)
        else:
            raise ValueError("The function name should be "
                             "`on_<eventname>`().")
        return event

    def _registered_func_name(self, event):
        funcNamelist = []
        for func in self._callbacks[event]:
            funcName = func.__module__ + '.' + func.__name__ + '_id' + str(id(func))
            funcNamelist.append(funcName)
        return funcNamelist

    def _create_emitter(self, event):
        """Create a method that emits an event of the same name."""
        if not hasattr(self, event):
            setattr(self, event,
                    lambda *args, **kwargs: self.emit(event, *args, **kwargs))

    def connect(self, func=None, event=None, set_method=False):
        """Register a callback function to a given event.
        To register a callback function to the `spam` event, where `obj` is
        an instance of a class deriving from `EventEmitter`:
        ```python
        @obj.connect
        def on_spam(arg1, arg2):
            pass
        ```
        This is called when `obj.emit('spam', arg1, arg2)` is called.
        Several callback functions can be registered for a given event.
        The registration order is conserved and may matter in applications.
        """
        if func is None:
            return partial(self.connect, set_method=set_method)

        # Get the event name from the function.
        if event is None:
            event = self._get_on_name(func)

        # We register the callback function.
        # if func is not in self._callbacks[event]:
        funcName = func.__module__ + '.' + func.__name__ + '_id' + str(id(func))
        if funcName not in self._registered_func_name(event):
            self._callbacks[event].append(func)

        # A new method self.event() emitting the event is created.
        if set_method:
            self._create_emitter(event)

        return func



    def unconnect(self, *funcs):
        """Unconnect specified callback functions."""
        for func in funcs:
            for callbacks in self._callbacks.values():
                if func in callbacks:
                    callbacks.remove(func)

    def emit(self, event, caller=None, *args, **kwargs):
        """Call all callback functions registered with an event.
        Any positional and keyword arguments can be passed here, and they will
        be forwarded to the callback functions.
        Return the list of callback return results.
        """
        res = []
        for callback in self._callbacks.get(event, []):
            if caller and caller == callback.__module__:
               continue 

            with Timer('[Event] emit -- {}'.format(callback.__module__), verbose=conf.ENABLE_PROFILER):
                res.append(callback(*args, **kwargs))
        return res

