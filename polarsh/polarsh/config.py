import sys
import platform
import numpy, scipy
import numba, quaternionic, spherical
import matplotlib, OpenGL, vispy, jupyter_rfb

def sys_info():
    out = 'Platform:     %s\n' % platform.platform()
    out += 'Python:       %s\n' % str(sys.version).replace('\n', ' ')
    out += '# [Numerical computation] ==============\n'
    out += 'NumPy:        %s\n' % numpy.__version__
    out += 'SciPy:        %s\n' % scipy.__version__
    out += 'Numba:        %s\n' % numba.__version__
    out += 'quaternionic: %s\n' % quaternionic.__version__
    out += 'spherical:    %s\n' % spherical.__version__
    out += '# [Visualization] ======================\n'
    out += 'Matplotlib:   %s\n' % matplotlib.__version__
    out += 'PyOpenGL:     %s\n' % OpenGL.__version__
    out += 'VisPy:        %s\n' % vispy.__version__
    out += 'jupyter_rfb:  %s\n' % jupyter_rfb.__version__

    return out