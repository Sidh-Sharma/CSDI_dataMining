import argparse
import torch
import datetime
import json
import yaml
import os

from main_model import CSDI_Traffic
from dataset_traffic import get_dataloader
from utils import train, evaluate

parser = argparse.ArgumentParser(description="CSDI Traffic")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for training/eval')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.1)
parser.add_argument(
    "--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])"
)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--use_physics", action="store_true", help="Enable dataset physics loss")
parser.add_argument("--lambda_phys", type=float, default=1e-6, help="Weight for physics loss term")

args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/traffic_fold" + str(args.nfold) + "_" + current_time + "/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

train_loader, valid_loader, test_loader, mean, std = get_dataloader(
    batch_size=config["train"]["batch_size"], 
    seed=args.seed, 
    missing_ratio=config["model"]["test_missing_ratio"]
)


model = CSDI_Traffic(
    config,
    args.device,
    use_physics=args.use_physics,
    lambda_phys=args.lambda_phys,
    mean=mean,
    std=std,
).to(args.device)

if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)
