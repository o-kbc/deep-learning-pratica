import torch

from torchvision.models import efficientnet

BASES_EFFICIENTNET = [
    "efficientnet_b0", "efficientnet_b1",
    "efficientnet_b2","efficientnet_b3","efficientnet_b4",
    "efficientnet_b5","efficientnet_b6","efficientnet_b7",
    "efficientnet_v2_l", 
    "efficientnet_v2_m", "efficientnet_v2_s"]

IMAGENET_WEIGHT = "EfficientNet_{}_Weights.IMAGENET1K_V1"

def get_base_from_torch(name, weights=None):
    if weights is not None:
        return eval(f"efficientnet.{name}(weights='{weights}')")
    else:
        return eval(f"efficientnet.{name}()")

class EfficientNet(torch.nn.Module):

    def __init__(self, output_size, 
                 base_model="efficientnet_b0",
                 pretrained=True,
                 dropout=.2):
        
        super().__init__()

        assert base_model in BASES_EFFICIENTNET

        self.base_name = base_model
        self.version   = base_model.removeprefix("efficientnet_")
        
        weights2load = IMAGENET_WEIGHT.format(self.version.upper())\
                        if pretrained else None

        self.base = get_base_from_torch(base_model, 
                        weights=weights2load).features
        
        self.pool = torch.nn.AdaptiveAvgPool2d(output_size=1)

        last_feat_size = self.base[-1][0].weight.shape[0]
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout, inplace=True),
            torch.nn.Linear(in_features=last_feat_size, 
                            out_features=output_size, bias=True)
        )
    
    def forward(self, x):

        x = self.base(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def get_base_name(self): return self.base_name
    def get_model_version(self): return self.version


if __name__ == "__main__":

    model = EfficientNet(10, "efficientnet_b0", pretrained=True, dropout=.2)
    print(model)
    dummy_input = torch.rand((1,3,224,224))
    print(model(dummy_input))

    model = EfficientNet(10, "efficientnet_b4", pretrained=True, dropout=.2)
    print(model)
    dummy_input = torch.rand((1,3,256,256))
    print(model(dummy_input))

    model = EfficientNet(10, "efficientnet_v2_s", pretrained=True, dropout=.2)
    print(model)
    dummy_input = torch.rand((1,3,480,480))
    print(model(dummy_input))