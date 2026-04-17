from pathlib import Path
import sys
import torch

if __name__ == "__main__" and __package__ is None:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

from experiments.phase_06_layernorm_residual_ffn.task_03_feed_forward import ff_output_data
from experiments.phase_06_layernorm_residual_ffn.task_02_layer_norm import normed_output1
from experiments.phase_04_qkv_attention.task_01_make_qkv import d_model


residual_2_sum = ff_output_data + normed_output1

norm2_layer = torch.nn.LayerNorm(d_model)

final_encoder_output = norm2_layer(residual_2_sum)

print("Final Encoder Output Shape:", final_encoder_output.shape)