from common import *

#  https://github.com/bermanmaxim/jaccardSegment/blob/master/losses.py
#  https://discuss.pytorch.org/t/solved-what-is-the-correct-way-to-implement-custom-loss-function/3568/4
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, logits, targets):
        return self.nll_loss(F.log_softmax(logits), targets)

class BCELoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, logits, targets):
        probs        = F.sigmoid(logits)
        probs_flat   = probs.view (-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(probs_flat, targets_flat)


class SoftDiceLoss(nn.Module):
    def __init__(self):  #weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()


    def forward(self, logits, targets):

        probs = F.sigmoid(logits)
        num = targets.size(0)
        m1  = probs.view(num,-1)
        m2  = targets.view(num,-1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1)+1)
        score = 1- score.sum()/num
        return score



##  http://geek.csdn.net/news/detail/126833
class WeightedBCELoss2d(nn.Module):
    def __init__(self):
        super(WeightedBCELoss2d, self).__init__()

    def forward(self, logits, labels, weights):
        w = weights.view(-1)
        z = logits.view (-1)
        t = labels.view (-1)
        loss = w*z.clamp(min=0) - w*z*t + w*torch.log(1 + torch.exp(-z.abs()))
        loss = loss.sum()/(w.sum()+ 1e-12)
        return loss

class WeightedSoftDiceLoss(nn.Module):
    def __init__(self):
        super(WeightedSoftDiceLoss, self).__init__()

    def forward(self, logits, labels, weights):
        probs = F.sigmoid(logits)
        num   = labels.size(0)
        w     = (weights).view(num,-1)
        w2    = w*w
        m1    = (probs  ).view(num,-1)
        m2    = (labels ).view(num,-1)
        intersection = (m1 * m2)
        score = 2. * ((w2*intersection).sum(1)+1) / ((w2*m1).sum(1) + (w2*m2).sum(1)+1)
        score = 1 - score.sum()/num
        return score


#https://github.com/marvis/pytorch-yolo2/blob/master/FocalLoss.py
#https://github.com/unsky/focal-loss
class FocalLoss2d(nn.Module):

    def __init__(self, gamma=2, size_average=True):
        super(FocalLoss2d, self).__init__()

        self.gamma = gamma
        self.size_average = size_average


    def forward(self, logits, targets, class_weights=None, type='softmax'):
        targets = targets.view(-1, 1).long()

        if type=='sigmoid':
            if class_weights is None: class_weights =[0.5, 0.5]

            probs  = F.sigmoid(logits)
            probs  = probs.view(-1, 1)
            probs  = torch.cat((1-probs, probs), 1)
            selects = Variable(torch.FloatTensor(len(probs), 2).zero_()).cuda()
            selects.scatter_(1, targets, 1.)

        elif  type=='softmax':
            B,C,H,W = logits.size()
            if class_weights is None: class_weights =[1/C]*C

            logits  = logits.permute(0, 2, 3, 1).contiguous().view(-1, C)
            probs   = F.softmax(logits,1)
            selects = Variable(torch.FloatTensor(len(probs), C).zero_()).cuda()
            selects.scatter_(1, targets, 1.)

        class_weights = Variable(torch.FloatTensor(class_weights)).cuda().view(-1,1)
        weights = torch.gather(class_weights, 0, targets)


        probs      = (probs*selects).sum(1).view(-1,1)
        batch_loss = -weights*(torch.pow((1-probs), self.gamma))*probs.log()

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss
        return loss





def dice_loss(m1, m2, is_average=True):
    num = m1.size(0)
    m1  = m1.view(num,-1)
    m2  = m2.view(num,-1)
    intersection = (m1 * m2)
    scores = 2. * (intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1)+1)
    if is_average:
        score = scores.sum()/num
        return score
    else:
        return scores


def multi_loss(logits, labels):
    #l = BCELoss2d()(logits, labels)


    if 0:
        l = BCELoss2d()(logits, labels) + SoftDiceLoss()(logits, labels)

    #compute weights
    else:
        batch_size,C,H,W = labels.size()
        weights = Variable(torch.tensor.torch.ones(labels.size())).cuda()

        if 1: #use weights
            kernel_size = 5
            avg = F.avg_pool2d(labels,kernel_size=kernel_size,padding=kernel_size//2,stride=1)
            boundary = avg.ge(0.01) * avg.le(0.99)
            boundary = boundary.float()

            w0 = weights.sum()
            weights = weights + boundary*2
            w1 = weights.sum()
            weights = weights/w1*w0

        l = WeightedBCELoss2d()(logits, labels, weights) + \
            WeightedSoftDiceLoss()(logits, labels, weights)

    return l


#
#
#
#
#
#
#
# class SoftCrossEntroyLoss(nn.Module):
#     def __init__(self):
#         super(SoftCrossEntroyLoss, self).__init__()
#
#     def forward(self, logits, soft_labels):
#         #batch_size, num_classes =  logits.size()
#         # soft_labels = labels.view(-1,num_classes)
#         # logits      = logits.view(-1,num_classes)
#
#         logits = logits - logits.max()
#         log_sum_exp = torch.log(torch.sum(torch.exp(logits), 1))
#         loss = - (soft_labels*logits).sum(1) + log_sum_exp
#         loss = loss.mean()
#
#         return loss
#
#
#
# # loss, accuracy -------------------------
# def top_accuracy(probs, labels, top_k=(1,)):
#     """Computes the precision@k for the specified values of k"""
#
#     probs  = probs.data
#     labels = labels.data
#
#     max_k = max(top_k)
#     batch_size = labels.size(0)
#
#     values, indices = probs.topk(max_k, dim=1, largest=True,  sorted=True)
#     indices  = indices.t()
#     corrects = indices.eq(labels.view(1, -1).expand_as(indices))
#
#     accuracy = []
#     for k in top_k:
#         # https://stackoverflow.com/questions/509211/explain-slice-notation
#         # a[:end]      # items from the beginning through end-1
#         c = corrects[:k].view(-1).float().sum(0, keepdim=True)
#         accuracy.append(c.mul_(1. / batch_size))
#     return accuracy
#
#
# ## focal loss ## ---------------------------------------------------
# class CrossEntroyLoss(nn.Module):
#     def __init__(self):
#         super(CrossEntroyLoss, self).__init__()
#
#     def forward(self, logits, labels):
#         #batch_size, num_classes =  logits.size()
#         # labels = labels.view(-1,1)
#         # logits = logits.view(-1,num_classes)
#
#         max_logits  = logits.max()
#         log_sum_exp = torch.log(torch.sum(torch.exp(logits-max_logits), 1))
#         loss = log_sum_exp - logits.gather(dim=1, index=labels.view(-1,1)).view(-1) + max_logits
#         loss = loss.mean()
#
#         return loss
#
# ## https://github.com/unsky/focal-loss
# ## https://github.com/sciencefans/Focal-Loss
# ## https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/39951
#
# #  https://raberrytv.wordpress.com/2017/07/01/pytorch-kludges-to-ensure-numerical-stability/
# #  https://github.com/pytorch/pytorch/issues/1620
# class FocalLoss(nn.Module):
#     def __init__(self,gamma = 2, alpha=1.2):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#
#
#     def forward(self, logits, labels):
#         eps = 1e-7
#
#         # loss =  - np.power(1 - p, gamma) * np.log(p))
#         probs = F.softmax(logits)
#         probs = probs.gather(dim=1, index=labels.view(-1,1)).view(-1)
#         probs = torch.clamp(probs, min=eps, max=1-eps)
#
#         loss = -torch.pow(1-probs, self.gamma) *torch.log(probs)
#         loss = loss.mean()*self.alpha
#
#         return loss
#
#
#
#
# # https://arxiv.org/pdf/1511.05042.pdf
# class TalyorCrossEntroyLoss(nn.Module):
#     def __init__(self):
#         super(TalyorCrossEntroyLoss, self).__init__()
#
#     def forward(self, logits, labels):
#         #batch_size, num_classes =  logits.size()
#         # labels = labels.view(-1,1)
#         # logits = logits.view(-1,num_classes)
#
#         talyor_exp = 1 + logits + logits**2
#         loss = talyor_exp.gather(dim=1, index=labels.view(-1,1)).view(-1) /talyor_exp.sum(dim=1)
#         loss = loss.mean()
#
#         return loss
#
# # check #################################################################
# def run_check_focal_loss():
#     batch_size  = 64
#     num_classes = 15
#
#     logits = np.random.uniform(-2,2,size=(batch_size,num_classes))
#     labels = np.random.choice(num_classes,size=(batch_size))
#
#     logits = Variable(torch.from_numpy(logits)).cuda()
#     labels = Variable(torch.from_numpy(labels)).cuda()
#
#     focal_loss = FocalLoss(gamma = 2)
#     loss = focal_loss(logits, labels)
#     print (loss)
#
#
# def run_check_soft_cross_entropy_loss():
#     batch_size  = 64
#     num_classes = 15
#
#     logits = np.random.uniform(-2,2,size=(batch_size,num_classes))
#     soft_labels = np.random.uniform(-2,2,size=(batch_size,num_classes))
#
#     logits = Variable(torch.from_numpy(logits)).cuda()
#     soft_labels = Variable(torch.from_numpy(soft_labels)).cuda()
#     soft_labels = F.softmax(soft_labels,1)
#
#     soft_cross_entropy_loss = SoftCrossEntroyLoss()
#     loss = soft_cross_entropy_loss(logits, soft_labels)
#     print (loss)

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_soft_cross_entropy_loss()

    print('\nsucess!')