import torch

def compute_iou_per_class(pred, target, num_classes):
    """
    Args:
        pred: torch.tensor[N, H, W]
        target: torch.tensor[N, H, W]
        num_classes: the class number, including the background class
        
    Returns:
        intersection: (C,)
        union: (C,)
    """
    pred = pred.long()
    target = target.long()
    pred_one_hot = torch.nn.functional.one_hot(pred, num_classes=num_classes).permute(0, 3, 1, 2)  # (N, C, H, W)
    target_one_hot = torch.nn.functional.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2)  # (N, C, H, W)
    
    intersection = (pred_one_hot & target_one_hot).sum(dim=(0, 2, 3))  # (C,)
    union = (pred_one_hot | target_one_hot).sum(dim=(0, 2, 3))  # (C,)

    return intersection, union

def compute_fscore_per_class(pred, target, num_classes):
    """
    Calculating the F1-score of each class
    Args:
        pred: shape [N, H, W]
        target: shape [N, H, W]
        num_classes: the class number, including the background class
        
    Returns:
        tp:  
        fp: 
        fn:  
    """
    pred = pred.long()
    target = target.long()

    pred_one_hot = torch.nn.functional.one_hot(pred, num_classes=num_classes).permute(0, 3, 1, 2)  # (N, C, H, W)
    target_one_hot = torch.nn.functional.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2)  # (N, C, H, W)

    tp = (pred_one_hot & target_one_hot).sum(dim=(0, 2, 3)).float()  # (C,)
    fp = (pred_one_hot & ~target_one_hot).sum(dim=(0, 2, 3)).float()  # (C,)
    fn = (~pred_one_hot & target_one_hot).sum(dim=(0, 2, 3)).float()  # (C,)    
    
    return tp, fp, fn