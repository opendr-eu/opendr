from torch import nn, hub

class DETRModel(nn.Module):
    def __init__(self, num_classes, num_queries, backbone, pretrained):
        
        super(DETRModel,self).__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        
        if backbone == "resnet50":
            self.model = hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=pretrained, num_classes=num_classes)
        elif backbone == "resnet101":
            self.model = hub.load('facebookresearch/detr', 'detr_resnet101', pretrained=pretrained, num_classes=num_classes)
        else:
             raise UserWarning('Possible backbones are: resnet50 and resnet101')
               
        self.model.num_queries = self.num_queries
    
    def forward(self,images):
        return self.model(images)

