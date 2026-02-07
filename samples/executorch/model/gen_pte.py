import torch
import torchvision.models as models

from torch.export import export, ExportedProgram
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights  # Import weights for MobileNetV3-Small

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import EdgeProgramManager, to_edge_transform_and_lower
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config
# from torchao.quantization.quant_api import Int8DynActInt4WeightQuantizer
from torch.export import export, export_for_training
from torch.nn.attention import sdpa_kernel, SDPBackend

# for profiling and visualization
from executorch.devtools import generate_etrecord
from executorch.devtools.backend_debug import get_delegation_info
from executorch.exir.backend.utils import format_delegated_graph
from tabulate import tabulate

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pte", type=str, default="model.pte", help="Path to output the PTE file.")
parser.add_argument("--model", type=str, choices=["mobilenet", "mobilenetv3small", "swin", "vit", "transformer"],
                    default="mobilenet",
                    help="Choose the model to export: 'mobilenet' (default), 'mobilenetv3small', 'swin', 'vit', or 'transformer'.")
parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "qs8", "qd8"],
                    default="qd8",
                    help="Choose the model data type, fp32 or fp16 or qs8 or qd8")
args = parser.parse_args()
pte_path = args.pte

print("Selected Model:", args.model)
print("Selected Precision:", args.precision)

if args.model == "mobilenet":
    model = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
    sample_inputs = (torch.randn(1, 3, 224, 224),)
elif args.model == "mobilenetv3small":
    model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).eval()
    sample_inputs = (torch.randn(1, 3, 224, 224),)
elif args.model == "swin":
    model = models.swin_t(weights="DEFAULT").eval()
    sample_inputs = (torch.randn(1, 3, 224, 224),)
elif args.model == "vit":
    model = models.vit_b_16(weights="DEFAULT").eval()
    # model = models.vit_b_32(weights="DEFAULT").eval()
    # model = models.vit_l_16(weights="DEFAULT").eval()
    # model = models.vit_l_32(weights="DEFAULT").eval()
    # model = models.vit_h_14(weights="DEFAULT").eval()
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

# half precision
if args.precision == "fp16":
    model = model.half()
    sample_inputs = (sample_inputs[0].half(),)
elif args.precision == "qs8":
    qparams = get_symmetric_quantization_config(is_per_channel=True) 
    quantizer = XNNPACKQuantizer()
    quantizer.set_global(qparams)
    training_ep = export(model, sample_inputs).module() 
    model = prepare_pt2e(training_ep, quantizer) 
    for cal_sample in sample_inputs:
        model(cal_sample) # (4) Calibrate
    model = convert_pt2e(model) # (5)
elif args.precision == "qd8":
    qparams = get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True) 
    quantizer = XNNPACKQuantizer()
    quantizer.set_global(qparams)
    training_ep = export(model, sample_inputs).module() 
    model = prepare_pt2e(training_ep, quantizer) 
    for cal_sample in sample_inputs:
        model(cal_sample) # (4) Calibrate
    model = convert_pt2e(model) # (5)

exported_program: ExportedProgram = export(model, sample_inputs)
edge_prog = to_edge_transform_and_lower(
    exported_program,
    partitioner=[XnnpackPartitioner()]
)
exec_prog = edge_prog.to_executorch()

with open(pte_path, "wb") as file:
    exec_prog.write_to_file(file)
# generate_etrecord(pte_path.replace(".pte", ".etrecord"), edge_prog, exec_prog)

graph_module = edge_prog.exported_program().graph_module
# print(format_delegated_graph(graph_module))
delegation_info = get_delegation_info(graph_module)
print(delegation_info.get_summary())
df = delegation_info.get_operator_delegation_dataframe()
print(tabulate(df, headers="keys", tablefmt="fancy_grid"))
