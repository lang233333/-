from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):  # 输入权重w，数据集x，数据集的标签y，正则化强度reg
    """
    Softmax loss function, naive implementation (with loops)
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
    Returns a tuple of:  # 返回损失和梯度
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #先做前馈过程的损失计算
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = np.dot(X, W)
    scores -= np.max(scores, axis = 1).reshape(num_train, 1)
    for i in range(num_train):
        loss += -np.log(np.exp(scores[i, y[i]]) / np.sum(np.exp(scores[i])))
        # 标签类的得分的幂次方除以所有类别得分的幂次方，外面再套一个-log（）
        # np.log是ln（）运算；loss是个累加的过程，它初始化是0，每张图片的loss都累加一下
        dW[:, y[i]] += (np.exp(scores[i, y[i]]) / np.sum(np.exp(scores[i])) - 1) * X[i]
        for j in range(num_classes):
            if j == y[i]:
                continue
            dW[:, j] += (np.exp(scores[i, j]) / np.sum(np.exp(scores[i]))) * X[i]
    loss /= num_train
    loss += reg * np.sum(W * W)
    
    dW /= num_train
    dW += reg * 2 * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #求梯度的时候严格按照反向传播的思想去做
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]  # 求一个矩阵或向量的梯度，一定要首先构建它的shape，本身的shape和梯度的shape是一样的
    scores = np.dot(X, W)  # np.dot()函数主要有两个功能,向量点积和矩阵乘法 shape是长度为c的向量 scores是n行c列的矩阵
    scores -= np.max(scores, axis = 1).reshape(num_train, 1)  #避免向量爆炸，因为是exp函数，每个样本减去样本里面的最大值，这样处理最大值就为0
    normalized_scores = np.exp(scores) / np.sum(np.exp(scores), axis = 1).reshape(num_train, 1)
    loss = -np.sum(np.log(normalized_scores[np.arange(num_train), y]))  # 标签类的得分的幂次方除以所有类别得分的幂次方，外面再套一个-log（）
    loss /= num_train
    loss += reg * np.sum(W * W)

    normalized_scores[np.arange(num_train), y] -= 1
    dW = np.dot(X.T, normalized_scores)  # 有n个样本就会有n个dw相加在一起
    dW /= num_train
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW