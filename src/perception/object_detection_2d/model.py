from torch import nn, hub
from torch.utils import AverageMeter

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

def train_fn(data_loader,model,criterion,optimizer,device,scheduler,epoch):
    
    model.train()
    criterion.train()
    
    summary_loss = AverageMeter()
    
    
    for step, (images, targets, image_ids) in enumerate(tk0):
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        

        output = model(images)
        
        loss_dict = criterion(output, targets)
        weight_dict = criterion.weight_dict
        
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        optimizer.zero_grad()

        losses.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        summary_loss.update(losses.item(),BATCH_SIZE)
        tk0.set_postfix(loss=summary_loss.avg)
        
    return summary_loss