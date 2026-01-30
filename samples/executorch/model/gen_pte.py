import torch
import torchvision.models as models

from torch.export import export, ExportedProgram
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from torchvision.models.squeezenet import SqueezeNet1_0_Weights  # Import weights for SqueezeNet
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights  # Import weights for MobileNetV3-Small

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import EdgeProgramManager, to_edge_transform_and_lower
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config

# from torchao.quantization.quant_api import Int8DynActInt4WeightQuantizer

# from model import GPT
from torch.export import export, export_for_training
from torch.nn.attention import sdpa_kernel, SDPBackend


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pte", type=str, default="model.pte", help="Path to output the PTE file.")
parser.add_argument("--model", type=str, choices=["mobilenet", "squeezenet", "lenet", "alexnet", "mobilenetv3small", "transformer"],
                    default="mobilenet",
                    help="Choose the model to export: 'mobilenet' (default), 'squeezenet', 'lenet', 'alexnet', 'mobilenetv3small', or 'transformer'.")
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
elif args.model == "squeezenet":
    model = models.squeezenet1_0(weights=SqueezeNet1_0_Weights.DEFAULT).eval()
    sample_inputs = (torch.randn(1, 3, 224, 224),)
elif args.model == "alexnet":
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT).eval()
    sample_inputs = (torch.randn(1, 3, 224, 224),)
elif args.model == "lenet":
    # Define a simple LeNet architecture.
    class LeNet(torch.nn.Module):
        def __init__(self):
            super(LeNet, self).__init__()
            self.conv1 = torch.nn.Conv2d(1, 6, kernel_size=5, stride=1)
            self.pool = torch.nn.MaxPool2d(2, 2)
            self.conv2 = torch.nn.Conv2d(6, 16, kernel_size=5, stride=1)
            self.fc1 = torch.nn.Linear(16 * 4 * 4, 120)
            self.fc2 = torch.nn.Linear(120, 84)
            self.fc3 = torch.nn.Linear(84, 10)
        def forward(self, x):
            x = torch.nn.functional.relu(self.conv1(x))
            x = self.pool(x)
            x = torch.nn.functional.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = torch.nn.functional.relu(self.fc1(x))
            x = torch.nn.functional.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    model = LeNet().eval()
    # LeNet expects a single-channel 28x28 image.
    sample_inputs = (torch.randn(1, 1, 28, 28),)
elif args.model == "mobilenetv3small":
    model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).eval()
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
    example_inputs = (torch.randint(0, 100, (1, model.config.block_size), dtype=torch.long), )
    dynamic_shape = (
        {1: torch.export.Dim("token_dim", max=model.config.block_size)},
        )

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
exec_prog = to_edge_transform_and_lower(
    exported_program,
    partitioner=[XnnpackPartitioner()]
).to_executorch()

with open(pte_path, "wb") as file:
    exec_prog.write_to_file(file)
