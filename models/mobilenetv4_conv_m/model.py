"""MobileNetV4-Conv-Medium (timm `mobilenetv4_conv_medium`), TFLite full-integer INT8 (MLPerf Mobile
recipe, see recipe.py). Shared code: models/mobilenetv4."""

from models.mobilenetv4.model_lib import (build, example_inputs, init_variables,  # noqa: F401
                                          input_spec, intermediates)
from models.mobilenetv4.net import CONV_MEDIUM_ARCH, Config

NAME = "mobilenetv4_conv_m"

# Reduced: every block kind of the full arch (er, uir ExtraDW/FFN/ConvNeXt, cn head
# block) plus an IR+dw_end UIR (the 'p' option, unused by the full arch); small
# widths, 1-4 blocks per stage, 64x64 input.
REDUCED_ARCH = (
    ('er_r1_k3_s2_e4_c16',),
    ('uir_r1_a3_k5_s2_e4_c24', 'uir_r1_a3_k3_s1_e2_c24'),
    ('uir_r1_a3_k5_s2_e6_c32', 'uir_r1_a0_k0_s1_e2_c32', 'uir_r1_a3_k0_s1_e4_c32',
     'uir_r1_a0_k3_s1_e2_c32_p3'),
    ('uir_r1_a5_k5_s2_e6_c48', 'uir_r1_a0_k0_s1_e4_c48', 'uir_r1_a5_k0_s1_e2_c48'),
    ('cn_r1_k1_s1_c96',),
)

CONFIGS = {
    "full": Config(name=NAME, arch_def=CONV_MEDIUM_ARCH, timm_name="mobilenetv4_conv_medium",
                   stem_size=32, num_features=1280, num_classes=1000,
                   layer_scale_init_value=None, image=256, batch=1),
    "reduced": Config(name=NAME, arch_def=REDUCED_ARCH, timm_name=None, stem_size=16,
                      num_features=128, num_classes=10, layer_scale_init_value=None,
                      image=64, batch=2),
}
