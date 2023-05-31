if __package__ is None or __package__ == '':
    # uses current directory visibility (running as script / jupyter notebook)
    from pam.pam_ops import *
    from pam.pam_opt import *
    import pam_experimental
else:
    # uses current package visibility (running as a module)
    from .pam_ops import *
    from .pam_opt import *
    from . import pam_experimental
