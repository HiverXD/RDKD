# src/model/load_teacher.py
import torch
from torchvision import models as tv

def make_resnet(name: str, num_classes: int):
    if name == "resnet18":
        m = tv.resnet18(weights=None)
    elif name == "resnet34":
        m = tv.resnet34(weights=None)
    else:
        raise ValueError(name)
    m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
    return m

def load_backbone_and_classifier(backbone_path, classifier_path, name, num_classes, map_location="cpu"):
    m = make_resnet(name, num_classes)
    # backbone(FC 제외) 로드
    bb = torch.load(backbone_path, map_location=map_location)
    m.load_state_dict({k: v for k, v in bb.items() if not k.startswith("fc.")}, strict=False)
    # classifier 로드
    cls = torch.load(classifier_path, map_location=map_location)
    m.load_state_dict({("fc." + k if not k.startswith("fc.") else k): v for k, v in cls.items()}, strict=False)
    return m

def load_student_imagenet(resnet18_pt_path, num_classes, map_location="cpu"):
    # 이미지넷 프리트레인 가중치(사용자 제공 .pt) 로드, FC는 새로 바인딩
    m = make_resnet("resnet18", num_classes)
    sd = torch.load(resnet18_pt_path, map_location=map_location)
    m.load_state_dict({k: v for k, v in sd.items() if "fc." not in k}, strict=False)
    m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
    return m
