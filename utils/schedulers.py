from torch.optim.lr_scheduler import LambdaLR

def lambda_rule(epoch):
    lr_l = 1.0 - max(0, epoch + 1 - 100) / float(100 + 1)
    return lr_l

def create_schedulers(optimizer_G, optimizer_D):
    scheduler_G = LambdaLR(optimizer_G, lr_lambda=lambda_rule)
    scheduler_D = LambdaLR(optimizer_D, lr_lambda=lambda_rule)
    return scheduler_G, scheduler_D
