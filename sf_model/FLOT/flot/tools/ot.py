import torch

"""
这个文件定义了一个名为sinkhorn的函数,该函数实现了Sinkhorn算法,用于计算点云1和点云2之间的最优传输计划. 

函数首先计算点云1和点云2中所有点之间的平方欧几里得距离. 这是通过计算点云1中每个点的平方和,然后添加点云2中每个点的平方和(转置操作用于保持矩阵维度一致),最后从结果中减去两次点云1和点云2的矩阵乘积来实现的. 

接下来,函数通过将距离小于100(即10平方)的位置设置为1,其他位置设置为0,来生成一个掩码矩阵(我们称之为support).

然后,函数计算传输成本矩阵C. 它首先对特征向量进行归一化(除以其L2范数),然后使用点积计算特征向量之间的相似性,并通过1减去这个相似性来获得成本. 

然后,函数计算指数化的成本矩阵K,这是通过将成本矩阵C除以正则化参数epsilon,然后取负数并计算指数结果来得到的. 最后,它将这个结果与支持矩阵相乘,以便在远于10米的点之间强制传输为零. 

在这之后,函数初始化Sinkhorn算法. 它首先计算一个权重power,然后初始化三个向量: a,prob1和prob2,它们都被设置为1除以它们各自的维度. 这些向量将在Sinkhorn迭代中被更新. 

Sinkhorn迭代会更新a和b向量,使用prob1和prob2向量来规范化它们的值. 这个过程会迭代max_iter次. 

最后,函数计算并返回传输矩阵T,它是通过将a和K的元素相乘,然后再与b的转置相乘得到的. 这个传输矩阵表示了从点云1到点云2的最优传输计划
"""

def sinkhorn(feature1, feature2, pcloud1, pcloud2, epsilon, gamma, max_iter):
    """
    Sinkhorn algorithm

    Parameters
    ----------
    feature1 : torch.Tensor
        Feature for points cloud 1. Used to computed transport cost. 
        Size B x N x C.
    feature2 : torch.Tensor
        Feature for points cloud 2. Used to computed transport cost. 
        Size B x M x C.
    pcloud1 : torch.Tensor
        Point cloud 1. Size B x N x 3.
    pcloud2 : torch.Tensor
        Point cloud 2. Size B x M x 3.
    epsilon : torch.Tensor
        Entropic regularisation. Scalar.
    gamma : torch.Tensor
        Mass regularisation. Scalar.
    max_iter : int
        Number of unrolled iteration of the Sinkhorn algorithm.

    Returns
    -------
    torch.Tensor
        Transport plan between point cloud 1 and 2. Size B x N x M.
    """

    # Squared l2 distance between points points of both point clouds
    distance_matrix = torch.sum(pcloud1 ** 2, -1, keepdim=True)
    distance_matrix = distance_matrix + torch.sum(
        pcloud2 ** 2, -1, keepdim=True
    ).transpose(1, 2)
    distance_matrix = distance_matrix - 2 * torch.bmm(pcloud1, pcloud2.transpose(1, 2))
    # Force transport to be zero for points further than 10 m apart
    support = (distance_matrix < 10 ** 2).float()

    # Transport cost matrix
    feature1 = feature1 / torch.sqrt(torch.sum(feature1 ** 2, -1, keepdim=True) + 1e-8)
    feature2 = feature2 / torch.sqrt(torch.sum(feature2 ** 2, -1, keepdim=True) + 1e-8)
    C = 1.0 - torch.bmm(feature1, feature2.transpose(1, 2))

    # Entropic regularisation
    K = torch.exp(-C / epsilon) * support

    # Early return if no iteration (FLOT_0)
    if max_iter == 0:
        return K

    # Init. of Sinkhorn algorithm
    power = gamma / (gamma + epsilon)
    a = (
        torch.ones(
            (K.shape[0], K.shape[1], 1), device=feature1.device, dtype=feature1.dtype
        )
        / K.shape[1]
    )
    prob1 = (
        torch.ones(
            (K.shape[0], K.shape[1], 1), device=feature1.device, dtype=feature1.dtype
        )
        / K.shape[1]
    )
    prob2 = (
        torch.ones(
            (K.shape[0], K.shape[2], 1), device=feature2.device, dtype=feature2.dtype
        )
        / K.shape[2]
    )

    # Sinkhorn algorithm
    for _ in range(max_iter):
        # Update b
        KTa = torch.bmm(K.transpose(1, 2), a)
        b = torch.pow(prob2 / (KTa + 1e-8), power)
        # Update a
        Kb = torch.bmm(K, b)
        a = torch.pow(prob1 / (Kb + 1e-8), power)

    # Transportation map
    T = torch.mul(torch.mul(a, K), b.transpose(1, 2))

    return T
