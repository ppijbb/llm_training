"""
TRL과 호환되는 커스텀 Reward Functions

TRL의 GRPOTrainer와 호환되는 커스텀 보상 함수들을 제공합니다.
"""

import torch
import logging
from typing import Dict, Any, List, Callable
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseRewardFunction(ABC):
    """TRL GRPOTrainer와 호환되는 기본 보상 함수 클래스"""

    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """
        TRL GRPOTrainer와 호환되는 보상 함수 호출 형식

        Args:
            completions: 생성된 응답들의 리스트
            **kwargs: 추가 인자들 (solutions, prompts 등)

        Returns:
            각 completion에 대한 보상 점수 리스트
        """
        pass


class AccuracyRewardFunction(BaseRewardFunction):
    """정확성 기반 보상 함수"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("accuracy", config)
        self.correct_keywords = self.config.get("correct_keywords", ["correct", "right", "yes"])

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """정확성 기반 보상 계산"""
        rewards = []
        solutions = kwargs.get("solutions", [])

        for completion in completions:
            reward = 0.0
            # 정확한 키워드가 포함되어 있는지 확인
            if any(keyword in completion.lower() for keyword in self.correct_keywords):
                reward = 1.0
            elif solutions:
                # 정답과 유사한지 확인 (간단한 매칭)
                if any(sol.lower() in completion.lower() or completion.lower() in sol.lower()
                      for sol in solutions):
                    reward = 0.8

            rewards.append(reward)

        return rewards


class LengthRewardFunction(BaseRewardFunction):
    """길이 기반 보상 함수"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("length", config)
        self.optimal_length = self.config.get("optimal_length", 100)
        self.length_weight = self.config.get("length_weight", 0.1)

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """길이 기반 보상 계산"""
        rewards = []

        for completion in completions:
            length = len(completion)
            # 최적 길이와의 차이로 보상 계산 (음수 보상 가능)
            length_diff = abs(length - self.optimal_length)
            reward = max(0, self.length_weight * (self.optimal_length - length_diff))

            rewards.append(reward)

        return rewards


class CustomRewardFunction(BaseRewardFunction):
    """사용자 정의 보상 함수"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("custom", config)
        # 사용자 정의 보상 로직 설정
        self.reward_scale = self.config.get("reward_scale", 1.0)
        self.penalty_scale = self.config.get("penalty_scale", -1.0)

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """사용자 정의 보상 계산"""
        rewards = []

        for completion in completions:
            reward = 0.0

            # 기본적인 품질 체크 (예시)
            if len(completion.strip()) > 10:  # 충분한 길이
                reward += self.reward_scale * 0.5

            if not completion.strip().endswith('.'):  # 문장 완성도
                reward += self.penalty_scale * 0.2

            # 추가 커스텀 로직은 여기에 구현
            # 예: 특정 키워드 포함 확인, toxicity 체크 등

            rewards.append(max(0, reward))  # 음수 보상 방지

        return rewards


def create_reward_function(reward_type: str, config: Dict[str, Any] = None) -> BaseRewardFunction:
    """
    보상 함수 팩토리 함수

    Args:
        reward_type: 보상 함수 타입 ("accuracy", "length", "custom")
        config: 보상 함수 설정

    Returns:
        생성된 보상 함수 인스턴스
    """
    config = config or {}

    if reward_type == "accuracy":
        return AccuracyRewardFunction(config)
    elif reward_type == "length":
        return LengthRewardFunction(config)
    elif reward_type == "custom":
        return CustomRewardFunction(config)
    else:
        logger.warning(f"Unknown reward type: {reward_type}, using custom")
        return CustomRewardFunction(config)


def combine_reward_functions(reward_functions: List[BaseRewardFunction]) -> BaseRewardFunction:
    """
    여러 보상 함수를 결합하는 함수

    Args:
        reward_functions: 결합할 보상 함수 리스트

    Returns:
        결합된 보상 함수
    """
    class CombinedRewardFunction(BaseRewardFunction):
        def __init__(self, functions: List[BaseRewardFunction]):
            super().__init__("combined", {})
            self.functions = functions

        def __call__(self, completions: List[str], **kwargs) -> List[float]:
            # 각 보상 함수의 결과를 평균
            all_rewards = []
            for func in self.functions:
                rewards = func(completions, **kwargs)
                all_rewards.append(rewards)

            # 보상 평균 계산
            combined_rewards = []
            for i in range(len(completions)):
                avg_reward = sum(rewards[i] for rewards in all_rewards) / len(all_rewards)
                combined_rewards.append(avg_reward)

            return combined_rewards

    return CombinedRewardFunction(reward_functions)
