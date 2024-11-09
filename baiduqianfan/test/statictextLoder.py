class StaticLoader:
    # 静态属性存储 example_group 和 example_prompt
    example_group1 = [
        {"input": "人生到处知何似，应似飞鸿踏雪泥", "output1": "何应", "output2": "何雪"},
        {"input": "倚杖柴门外，临风听慕蝉", "output1": "临风", "output2": "慕蝉"},
        {"input": "竹喧归浣女，莲动下渔舟", "output1": "宣舟", "output2": "宣瑜"}
    ]

    example_prompt = "请根据古诗词再帮我起一个男孩名和一个女孩名"

    # 静态属性存储 text 列表
    text = [
        "经济学的核心概念之一。资源是有限的，而人类的欲望是无限的，经济学因此研究如何在稀缺资源的前提下做出选择。",
        "市场价格由供给和需求决定。当需求增加或供给减少时，价格会上升；反之亦然。",
        "凯恩斯主义主张政府干预市场，以应对经济衰退；而古典经济学认为市场自我调节的能力更强。",
        "经济学中用来分析个人或团体在战略环境下的决策。著名的囚徒困境便是博弈论的经典例子。",
        "机器学习分为监督学习和非监督学习。监督学习使用带有标签的数据来训练模型，常用于分类和回归问题；非监督学习则使用未标注的数据进行模式发现和聚类分析。",
        "回归是用于预测连续值的任务，如房价预测；分类是用于预测离散类别的任务，如垃圾邮件检测，常见的分类算法包括决策树、支持向量机和神经网络。",
        "神经网络模仿人脑的工作方式，深度学习是多层神经网络的一种，能够处理复杂的模式和特征，尤其在图像识别、自然语言处理等领域取得了巨大进展。",
        "在模型训练中，过拟合是指模型在训练数据上表现良好，但在新数据上表现不佳；欠拟合则是模型在训练数据上和新数据上均表现不佳。解决过拟合的常用方法包括正则化和增加训练数据。",
        "特征工程是将原始数据转换为适合机器学习模型的特征的过程，好的特征工程可以显著提高模型性能。常用的技术包括特征选择、标准化、归一化和特征组合。",
        "指价格普遍上涨，货币购买力下降。通胀过高会损害经济，但适度的通胀被认为是经济增长的标志。"
    ]
    # 示例组2
    example_group2 = "你是一个{name}，请模仿示例起3个{country}名字，比如男孩经常被叫做{boy},女孩经常被叫做{girl}"
    @staticmethod
    def get_example_group1():
        """返回 example_group 的内容"""
        return StaticLoader.example_group1

    @staticmethod
    def get_example_prompt():
        """返回 example_prompt 的内容"""
        return StaticLoader.example_prompt

    @staticmethod
    def get_text():
        """返回 text 的内容"""
        return StaticLoader.text
    @staticmethod
    def get_example_group2():
        return StaticLoader.example_group2