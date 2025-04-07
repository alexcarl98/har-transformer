import yaml
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List, Literal

import numpy as np
import pandas as pd
import torch

@dataclass
class Config:
    """Configuration for the experiment."""
    data_dir: str = "raw_data/"
    output_dir: str = "doc/latex/figure/"
