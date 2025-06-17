import warnings

import rich
import rich.traceback
from tqdm import TqdmExperimentalWarning

handler = rich.traceback.install(show_locals=False)
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
