# SPDX-License-Identifier: Apache-2.0
from typing import Dict, Type, Any

# Forward declaration to avoid circular imports
class BaseSteerVectorAlgorithm:
    pass

# 全局算法注册表
ALGORITHM_REGISTRY: Dict[str, Type["BaseSteerVectorAlgorithm"]] = {}


def register_algorithm(name: str):
    """
    一个装饰器，用于将算法类注册到全局注册表中。
    
    Args:
        name: 算法的唯一名称 (例如, "direct", "loreft").
    """
    def decorator(cls: Type["BaseSteerVectorAlgorithm"]):
        if name in ALGORITHM_REGISTRY:
            # 在实践中，根据需要可以换成更宽容的策略，例如 logging.warning
            raise ValueError(f"Algorithm '{name}' is already registered.")
        ALGORITHM_REGISTRY[name] = cls
        return cls
    return decorator


def create_algorithm(name: str, *args, **kwargs) -> "BaseSteerVectorAlgorithm":
    """
    算法工厂函数，根据名称创建算法实例。
    
    Args:
        name: 要创建的算法的名称。
        *args, **kwargs: 传递给算法构造函数的参数。
        
    Returns:
        一个BaseSteerVectorAlgorithm的实例。
        
    Raises:
        ValueError: 如果算法名称未被注册。
    """
    if name not in ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown algorithm: '{name}'. Available algorithms: {list(ALGORITHM_REGISTRY.keys())}")
    
    # 导入BaseSteerVectorAlgorithm的真实定义
    from .base import BaseSteerVectorAlgorithm as ConcreteBase
    
    cls = ALGORITHM_REGISTRY[name]
    
    # 确保返回的实例类型是正确的
    instance: ConcreteBase = cls(*args, **kwargs)
    return instance 