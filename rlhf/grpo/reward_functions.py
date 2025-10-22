"""
통합 커스텀 Reward Functions for TRL GRPO

단일 또는 다중 보상 함수를 지원하는 통합 보상 시스템입니다.
"""
from functools import wraps
import logging
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class RewardComponent(ABC):
    """보상 계산의 개별 컴포넌트"""

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight

    @abstractmethod
    def calculate(self, completion: str, **kwargs) -> float:
        """개별 보상 계산"""
        pass


class AccuracyComponent(RewardComponent):
    """정확성 보상 컴포넌트"""

    def __init__(self, config: Dict[str, Any] = None, weight: float = 1.0):
        super().__init__("accuracy", weight)
        self.correct_keywords = config.get("correct_keywords", ["correct", "right", "yes"]) if config else ["correct", "right", "yes"]

    def calculate(self, completion: str, **kwargs) -> float:
        solutions = kwargs.get("solutions", [])
        completion_lower = completion.lower()

        # 정확한 키워드가 포함되어 있는지 확인
        if any(keyword in completion_lower for keyword in self.correct_keywords):
            return 1.0
        elif solutions:
            # 정답과 유사한지 확인
            if any(sol.lower() in completion_lower or completion_lower in sol.lower()
                  for sol in solutions):
                return 0.8

        return 0.0


class LengthComponent(RewardComponent):
    """길이 보상 컴포넌트"""

    def __init__(self, config: Dict[str, Any] = None, weight: float = 1.0):
        super().__init__("length", weight)
        self.optimal_length = config.get("optimal_length", 100) if config else 100
        self.length_weight = config.get("length_weight", 0.1) if config else 0.1

    def calculate(self, completion: str, **kwargs) -> float:
        length = len(completion)
        length_diff = abs(length - self.optimal_length)
        return max(0, self.length_weight * (self.optimal_length - length_diff))


class QualityComponent(RewardComponent):
    """품질 보상 컴포넌트"""

    def __init__(self, config: Dict[str, Any] = None, weight: float = 1.0):
        super().__init__("quality", weight)
        self.reward_scale = config.get("reward_scale", 1.0) if config else 1.0
        self.penalty_scale = config.get("penalty_scale", -1.0) if config else -1.0
        self.quality_keywords = config.get("quality_keywords", ["좋아", "완벽", "우수"]) if config else ["좋아", "완벽", "우수"]
        self.negative_keywords = config.get("negative_keywords", ["모르겠", "잘못", "틀렸"]) if config else ["모르겠", "잘못", "틀렸"]

    def calculate(self, completion: str, **kwargs) -> float:
        reward = 0.0
        completion_lower = completion.lower()

        # 충분한 길이 체크
        if len(completion.strip()) > 10:
            reward += self.reward_scale * 0.5

        # 문장 완성도 체크
        if not completion.strip().endswith('.'):
            reward += self.penalty_scale * 0.2

        # 품질 키워드 체크
        if any(keyword in completion for keyword in self.quality_keywords):
            reward += self.reward_scale * 0.3

        # 부정 키워드 체크
        if any(keyword in completion_lower for keyword in self.negative_keywords):
            reward += self.penalty_scale * 0.3

        return max(0, reward)

class BaseRewardFunction:
    __name__ = "RewardFunction"

    @abstractmethod
    def reward_func(self, *args, **kwargs) -> List[float]:
        raise NotImplementedError("Subclasses must implement this method")

    def __call__(self, *args, **kwargs) -> List[float]:
        return self.reward_func(*args, **kwargs)

    def __str__(self):
        return self.__name__

    def __repr__(self):
        return self.__name__

class MultiRewardFunction(BaseRewardFunction):
    """
    다중 보상 함수를 지원하는 통합 클래스
    
    reward 처리
        completion 예시: [{role: "assistant", content: "어쩌고 저쩌고..."}]
        reward function에서는 위의 completion의 content를 처리하여 보상 부여.
    """

    def __init__(
        self,
        components: Optional[List[RewardComponent]] = None,
        config: Dict[str, Any] = None
    ):
        self.config = config or {}
        self.components = components or []

        # 기본 컴포넌트들 설정
        if not self.components:
            self.components = [
                AccuracyComponent(self.config.get("accuracy", {})),
                LengthComponent(self.config.get("length", {})),
                QualityComponent(self.config.get("quality", {}))
            ]

    def reward_func(self, completions: List[str], **kwargs) -> List[float]:
        """TRL GRPOTrainer와 호환되는 다중 보상 계산"""
        rewards = []

        for completion in completions:
            total_reward = 0.0

            # 각 컴포넌트의 보상을 가중합으로 계산
            for component in self.components:
                component_reward = component.calculate(completion, **kwargs)
                total_reward += component_reward * component.weight

            # 정규화 (최대 보상이 1.0이 되도록)
            max_possible_reward = sum(comp.weight for comp in self.components)
            if max_possible_reward > 0:
                total_reward = total_reward / max_possible_reward

            rewards.append(max(0, total_reward))  # 음수 보상 방지

        return rewards

    def add_component(self, component: RewardComponent):
        """새로운 보상 컴포넌트 추가"""
        self.components.append(component)
        logger.info(f"✅ Added reward component: {component.name}")

    def remove_component(self, name: str):
        """보상 컴포넌트 제거"""
        self.components = [comp for comp in self.components if comp.name != name]
        logger.info(f"✅ Removed reward component: {name}")


class SingleCustomRewardFunction(BaseRewardFunction):
    """
    단일 커스텀 보상 함수
    
    reward 처리
        completion 예시: [{role: "assistant", content: "어쩌고 저쩌고..."}]
        reward function에서는 위의 completion의 content를 처리하여 보상 부여.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # 통합된 보상 로직 설정
        self.accuracy_weight = self.config.get("accuracy_weight", 0.4)
        self.length_weight = self.config.get("length_weight", 0.2)
        self.quality_weight = self.config.get("quality_weight", 0.4)

        # 각 컴포넌트 설정
        self.accuracy_config = self.config.get("accuracy", {})
        self.length_config = self.config.get("length", {})
        self.quality_config = self.config.get("quality", {})

    def reward_func(self, completions: List[str], **kwargs) -> List[float]:
        """단일 통합 보상 계산"""
        rewards = []

        for completion in completions:
            reward = 0.0
            completion_content = completion[0]["content"]
            # 정확성 보상
            if self.accuracy_config.get("correct_keywords"):
                if any(keyword in completion_content.lower() for keyword in self.accuracy_config["correct_keywords"]):
                    reward += self.accuracy_weight * 1.0

            # 길이 보상
            optimal_length = self.length_config.get("optimal_length", 100)
            length_weight = self.length_config.get("length_weight", 0.1)
            length = len(completion_content)
            length_diff = abs(length - optimal_length)
            length_reward = max(0, length_weight * (optimal_length - length_diff))
            reward += self.length_weight * length_reward

            # 품질 보상
            quality_reward = 0.0
            if len(completion_content.strip()) > 10:
                quality_reward += self.quality_weight * 0.5

            if not completion_content.strip().endswith('.'):
                quality_reward += self.quality_weight * (-0.2)

            reward += quality_reward

            rewards.append(max(0, reward))

        return rewards


def create_reward_function(reward_type: str, config: Dict[str, Any] = None):
    """
    통합 보상 함수 팩토리 함수

    Args:
        reward_type: 보상 함수 타입 ("single", "multi", "accuracy", "length", "quality")
        config: 보상 함수 설정

    Returns:
        생성된 보상 함수 인스턴스
    """
    config = config or {}

    if reward_type == "single":
        return SingleCustomRewardFunction(config)
    elif reward_type == "multi":
        return MultiRewardFunction(config=config)
    elif reward_type == "accuracy":
        return AccuracyComponent(config)
    elif reward_type == "length":
        return LengthComponent(config)
    elif reward_type == "quality":
        return QualityComponent(config)
    else:
        logger.warning(f"Unknown reward type: {reward_type}, using single custom")
        return SingleCustomRewardFunction(config)


def create_multi_reward_function(config: Dict[str, Any] = None) -> MultiRewardFunction:
    """다중 보상 함수 생성 (기본 컴포넌트들 포함)"""
    return MultiRewardFunction(config=config)


def create_single_reward_function(config: Dict[str, Any] = None) -> SingleCustomRewardFunction:
    """단일 통합 보상 함수 생성"""
    return SingleCustomRewardFunction(config)
