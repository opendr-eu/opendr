import torch
from util.box_ops import rescale_bboxes

@torch.no_grad()
def detect(im, transform, model, device, threshold):
        # mean-std normalize the input image (batch-size: 1)
        img = transform(im).unsqueeze(0)

        # propagate through the mode
        img = img.to(torch.device(device))

        model.eval()
        outputs = model(img)

        # keep only predictions with 0.7+ confidence
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > threshold

        # convert boxes from [0; 1] to image scales
        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size, device)
        return probas[keep], bboxes_scaled
