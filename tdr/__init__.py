"""TDR Fault and Load Identification package."""

__version__ = "1.0.0"

from tdr.simulator import (
    LineParams,
    simulate_trace,
    simulate_batch,
    analytic_bounce_diagram,
    analytic_reactive_tail,
    compute_tnom,
    compute_fault_gamma_nominal,
    C_CONST,
)
from tdr.data import (
    FAULT_CLASSES,
    LOAD_CLASSES,
    FAULT_TO_ID,
    ID_TO_FAULT,
    LOAD_TO_ID,
    ID_TO_LOAD,
    preprocess,
    preprocess_numpy,
    TDRDataset,
    load_real_wise_casing,
)
from tdr.features import (
    ConventionalTDR,
    extract_features,
)
from tdr.models import (
    LGBMSuite,
    ResNet1D,
    MaskedMultiTaskLoss,
)
from tdr.metrics import (
    evaluate_t1_fault,
    evaluate_t2_loc,
    evaluate_t3_load,
    compute_all_metrics,
)
