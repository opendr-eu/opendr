import torch
from util.box_ops import rescale_bboxes
import numpy as np

@torch.no_grad()
def detect(im, transform, model, device, threshold, ort_session):
        dev = torch.device(device)        
    
        # mean-std normalize the input image (batch-size: 1)
        img = transform(im).unsqueeze(0)
        
        if ort_session is not None:
            # propagate through the onnx model
            outputs = ort_session.run(['pred_logits', 'pred_boxes'], {'data': np.array(img)})
            
            pred_logits = torch.tensor(outputs[0], device=dev)
            pred_boxes = torch.tensor(outputs[1], device=dev)
            
            # keep only predictions with threshold confidence
            probas = pred_logits.softmax(-1)[0, :, :-1]
            keep = probas.max(-1).values > threshold
            
            # convert boxes from [0; 1] to image scales
            bboxes_scaled = rescale_bboxes(pred_boxes[0, keep], im.size, device)
            return probas[keep], bboxes_scaled
        else:
            # propagate through the pytorch model
            img = img.to(dev)
            model.eval()
            outputs = model(img)

            # keep only predictions with threshold confidence
            probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
            keep = probas.max(-1).values > threshold
    
            # convert boxes from [0; 1] to image scales
            bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size, device)
            return probas[keep], bboxes_scaled
