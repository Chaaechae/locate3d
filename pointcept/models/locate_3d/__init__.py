# Locate-3D localization downstream on Utonia features.
#
# Locate3DSegDetector is the production model used by the kept configs
# (localize-utonia-v1m1-0h / 0i / 0j). It grounds a referring expression
# by predicting a per-point mask per entity and deriving an axis-aligned
# box from that mask. See pointcept/models/locate_3d/locate_3d_segdet.py.
#
# The earlier DETR-style set-prediction decoder (Locate3DLocalizer +
# matcher/criterion/decoder) was removed during cleanup -- it never
# converged on Utonia's raw 9-dim features. See LEARNINGS.md for the full
# post-mortem of why that path was abandoned in favour of segmentation.
from .locate_3d_segdet import Locate3DSegDetector
