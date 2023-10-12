import pynvml

def get_gpu_info(use_index=(0,)):
    """
    深度学习训练使用，可以回去显卡信息，
    使用到的包:pynvml
    :param use_index: 使用的GPU的物理编号
    :return: 
    """

    # 计算显存是GB还是MB的函数，方便后续查看数据
    def func(number):
        # number单位是MB
        if number // 1024 > 0:  # 如果number对1024取整是大于0的说明单位是GB
            return f"{number / 1024.0:.3f}GB"  # 返回值的单位是GB
        else:
            return f"{number:.3f}MB"

    # 初始化管理工具
    pynvml.nvmlInit()
    # device = torch.cuda.current_device()  # int
    gpu_count = pynvml.nvmlDeviceGetCount()  # int
    information = []
    for index in range(gpu_count):
        # 不是使用的gpu，就剔除
        if index not in use_index:
            continue
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total = meminfo.total / 1024 ** 2  # 总的显存大小,单位是MB
        used = meminfo.used / 1024 ** 2  # 已用显存大小
        free = meminfo.free / 1024 ** 2  # 剩余显存大小
        information.append(f"Memory Total:{func(total)}; Memory Used:{func(used)}; Memory Free:{func(free)}")
    # 关闭管理工具
    pynvml.nvmlShutdown()
    return free
    # return "\n".join(information)