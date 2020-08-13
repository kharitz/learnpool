
def bin_accuracy(output, target):
    """
    Computes the accuracy for multiple binary predictions

    """
    oo = output.clone()
    tt = target.clone()
    num = oo.eq(tt).sum().double().item()
    den = tt.numel()
    acc = 100 * num / den

    return acc
