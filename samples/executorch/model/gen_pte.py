import torch
import torchvision.models as models

from torch.export import export, ExportedProgram

from torchvision.models.squeezenet import SqueezeNet1_0_Weights  # Import weights for SqueezeNet
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights  # Import weights for MobileNetV3-Small
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.utils.configs import get_xnnpack_edge_compile_config
from executorch.exir import EdgeProgramManager, ExecutorchProgramManager, to_edge
from executorch.exir.backend.backend_api import to_backend

# from model import GPT
from torch.export import export, export_for_training
from torch.nn.attention import sdpa_kernel, SDPBackend


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pte", type=str, default="model.pte", help="Path to output the PTE file.")
parser.add_argument("--model", type=str, choices=["mobilenet", "squeezenet", "alexnet", "resnet", "mobilenetv3small", "mobilenetv3large", "transformer"],
                    default="mobilenet",
                    help="Choose the model to export: 'mobilenet' (default), 'squeezenet', 'alexnet', resnet, 'mobilenetv3small', 'mobilenetv3large', or 'transformer'.")
parser.add_argument("--precision", type=str, choices=["fp32", "fp16"],
                    default="fp32",
                    help="Choose the model data type, fp32 or fp16")
args = parser.parse_args()
pte_path = args.pte
print("Torch version:", torch.__version__)
print("Selected Model:", args.model)

if args.model == "squeezenet":
    model = models.squeezenet1_0(weights=SqueezeNet1_0_Weights.DEFAULT).eval()
    sample_inputs = (torch.randn(1, 3, 224, 224),)
elif args.model == "alexnet":
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT).eval()
    sample_inputs = (torch.randn(1, 3, 224, 224),)
elif args.model == "resnet":
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).eval()
    sample_inputs = (torch.randn(1, 3, 224, 224),)
elif args.model == "transformer":
    class SimpleTransformer(torch.nn.Module):
        def __init__(self, d_model=64, nhead=8, dim_feedforward=128, seq_len=16):
            super(SimpleTransformer, self).__init__()
            self.pos_embedding = torch.nn.Parameter(torch.randn(1, seq_len, d_model))
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
            self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        def forward(self, x):
            x = x + self.pos_embedding  # Add positional encoding
            return self.encoder(x)
    
    model = SimpleTransformer().eval()
    sample_inputs = (torch.randn(1, 16, 64),)  # (batch_size, seq_len, d_model)
elif args.model == "gpt2":
    model = GPT.from_pretrained('gpt2')
    sample_inputs = (torch.randint(0, 100, (1, model.config.block_size), dtype=torch.long), )
    dynamic_shape = (
        {1: torch.export.Dim("token_dim", max=model.config.block_size)},
        )
elif args.model == "mobilenetv3small":
    model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).eval()
    sample_inputs = (torch.randn(1, 3, 224, 224),)
elif args.model == "mobilenetv3large":
    model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT).eval()
    sample_inputs = (torch.randn(1, 3, 224, 224),)
elif args.model == "mobilenet":
    model = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
    sample_inputs = (torch.randn(1, 3, 224, 224),)

from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
  XNNPACKQuantizer,
  get_symmetric_quantization_config,
)
quantizer = XNNPACKQuantizer()
quantizer.set_global(get_symmetric_quantization_config())
exported_model = export_for_training(model, sample_inputs).module()
prepared_model = prepare_pt2e(exported_model, quantizer)
quantized_model = convert_pt2e(prepared_model)
exported_program: ExportedProgram = export(quantized_model, sample_inputs)
edge: EdgeProgramManager = to_edge(exported_program)

# Set up the partitioner (using XnnpackPartitioner as in the original code)
edge = edge.to_backend(XnnpackPartitioner())

exec_prog = edge.to_executorch()

with open(pte_path, "wb") as file:
    exec_prog.write_to_file(file)
