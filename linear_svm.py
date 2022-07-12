from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
    Inputs:
    - W: A numpy array of shape (D, C) containing weights.  # W是权重矩阵，是D行C列的
    - X: A numpy array of shape (N, D) containing a minibatch of data.  # X是输入数据，是N行D列的
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means  # Y里面每个数字代表训练数据对应标签，是N行的
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength  # 是正则化的参数
    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W  # 对W的梯度，大小与W矩阵相同
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]  # 类别数
    num_train = X.shape[0]  # 图片数
    loss = 0.0  # 损失初始化为零
    for i in range(num_train):  # 这个i是对所有数据集的图片进行一个遍历
        scores = X[i].dot(W)  # scores.shapes(C,) 这张图片每个类别的得分
        correct_class_score = scores[y[i]]  # 标签类得分
        for j in range(num_classes):  #j是每个W有多少线性分类器，有多少个类别，进行遍历
            if j == y[i]:   #如果它遍历到标签位的时候，就不做累加，因为只有非标签类才计入损失（跳过标签类）
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:  # 判断条件是margin成立时
                loss += margin  
                dW[:,j] += X[i]  # 非标签类，里面累加的是X[i]
                dW[:,y[i]] -= X[i]  
            # 这就是标签类，在对他进行一个梯度的累加的时候，y[i]里面给的是标签索引，dW每次对括号内的东西做一个累加，累加的东西是-X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                               #不需要把两部分代码分开来做，可以将反向求导代码在原有基础上修改
    # Compute the gradient of the loss function and store it dW.            #将W的梯度传到dW里面去
    # Rather that first computing the loss and then computing the derivative,   #不需要求出损失再计算梯度
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW /= num_train  # 除以N（训练数据的多少）别忘了
    dW += reg * 2 * W  # 正则化损失项的导数（正则化损失项是αR（W）即αW²，求导就是2αW）

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.
    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_classes = W.shape[1]  # 计算类别数
    num_train = X.shape[0]  # 计算样本数
    scores = np.dot(X, W)  # 计算scores矩阵，scores.shape=(N,10)
    correct_class_score = scores[np.arange(num_train), y].reshape(num_train, 1)  # 从scores矩阵中拿出它标签类的得分，reshape成（n，1）为了与上做广播
    margin = np.maximum(0, scores - correct_class_score + 1)  # ＋delta
    margin[np.arange(num_train), y] = 0  # 加完delta后，强制把标签类置为零
    loss = np.sum(margin) / num_train + reg * np.sum(W ** 2)  # 求和再除以N（num_train张图片），这一部分是数据损失，后一部分是正则化损失

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    margin[margin > 0] = 1
    correct_number = np.sum(margin, axis = 1)  # 统计有多少个1（多少个margin），就减去多少次
    margin[np.arange(num_train), y] -= correct_number
    dW = np.dot(X.T, margin) / num_train + reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW