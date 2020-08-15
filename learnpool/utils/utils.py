
def bin_accuracy(output, target):
    """
    Computes the accuracy for multiple binary predictions
    Args:   
    output --> Class prediction
    target --> Class label
    Returns:
    acc --> Binary classification accuracy 

    """
    oo = output.clone()
    tt = target.clone()
    num = oo.eq(tt).sum().double().item()
    den = tt.numel()
    acc = 100 * num / den

    return acc
