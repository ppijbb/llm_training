"""
통합 커스텀 Reward Functions for TRL GRPO

단일 또는 다중 보상 함수를 지원하는 통합 보상 시스템입니다.
"""
import logging
import re
from typing import Dict, Any, List, Set
from reward.reward_functions import RewardComponent, AccuracyComponent, LengthComponent, QualityComponent
from reward.reward_functions import MultiRewardFunction, SingleCustomRewardFunction


logger = logging.getLogger(__name__)


class ToothNumberComponent(RewardComponent):
    """
    치아 번호 매칭 (F1 + 엄격한 페널티)
    가중치: 0.50
    """
    
    def __init__(self, weight: float = 0.50):
        super().__init__(name="tooth_number", weight=weight)
        
        # 엄격한 설정
        self.perfect_match_bonus = 0.2  # 완벽 매칭 시 보너스
        self.hallucination_penalty = 0.4  # 없는 치아 추가 시 감점
        self.missing_penalty = 0.3  # 치아 누락 시 감점
        self.wrong_tooth_penalty = 0.5  # 완전히 다른 치아 시 감점
    
    def calculate(self, completion: str, ground_truth: str = None, **kwargs) -> float:
        """
        엄격한 치아 번호 평가
        
        Examples:
            GT: "number 7, 8, 9"
            
            Gen1: "number 7, 8, 9" 
                → Perfect! F1=1.0 + bonus=0.2 → 1.2 (clipped to 1.0)
            
            Gen2: "number 7, 8"
                → Missing 9: F1=0.8 - missing_penalty(0.3) → 0.5
            
            Gen3: "number 7, 8, 9, 10"
                → Hallucination: F1=0.86 - hallucination_penalty(0.4) → 0.46
            
            Gen4: "number 1, 2, 3"
                → Completely wrong: F1=0.0 - wrong_tooth_penalty(0.5) → -0.5 (clipped to 0.0)
        """
        gen_teeth = self._extract_teeth_strict(completion)
        gt_teeth = self._extract_teeth_strict(ground_truth)
        
        if not gt_teeth:
            # Ground truth에 치아 번호가 없는 경우
            return 1.0 if not gen_teeth else 0.3
        
        if not gen_teeth:
            # 치아 번호를 아예 생성하지 않음 → 심각한 오류
            return 0.0
        
        gen_set = set(gen_teeth)
        gt_set = set(gt_teeth)
        
        # 정확한 매칭 계산
        correct = gen_set & gt_set  # True Positives
        hallucinated = gen_set - gt_set  # False Positives
        missing = gt_set - gen_set  # False Negatives
        
        # F1 Score
        if len(gen_set) == 0:
            precision = 0.0
        else:
            precision = len(correct) / len(gen_set)
        
        if len(gt_set) == 0:
            recall = 0.0
        else:
            recall = len(correct) / len(gt_set)
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        # 기본 점수
        score = f1
        
        # 완벽 매칭 보너스
        if gen_set == gt_set:
            score += self.perfect_match_bonus
        
        # Hallucination 페널티 (없는 치아 추가)
        if hallucinated:
            hallucination_ratio = len(hallucinated) / len(gt_set)
            score -= self.hallucination_penalty * hallucination_ratio
            
            # 매우 심각한 hallucination (2배 이상 추가)
            if len(hallucinated) >= len(gt_set):
                score -= 0.2
        
        # 누락 페널티
        if missing:
            missing_ratio = len(missing) / len(gt_set)
            score -= self.missing_penalty * missing_ratio
        
        # 완전히 틀린 경우 강력한 페널티
        if len(correct) == 0:
            score -= self.wrong_tooth_penalty
        
        # 0.0 ~ 1.0 범위로 클리핑
        return max(0.0, min(1.0, score))
    
    def _extract_teeth_strict(self, text: str) -> List[int]:
        """
        엄격한 치아 번호 추출
        - number X 패턴만 인정
        - repeat X는 치아 번호가 아닌 경우도 있으므로 컨텍스트 분석
        """
        teeth = []
        
        # 1) "number X" 패턴 (단일)
        single_pattern = r'\bnumber\s+(\d+)\b'
        for match in re.finditer(single_pattern, text, flags=re.IGNORECASE):
            tooth_num = int(match.group(1))
            if 1 <= tooth_num <= 32 or 11 <= tooth_num <= 48:  # UNS or FDI 범위
                teeth.append(tooth_num)
        
        # 2) "number X-Y" 패턴 (범위)
        range_pattern = r'\bnumber\s+(\d+)\s*-\s*(\d+)\b'
        for match in re.finditer(range_pattern, text, flags=re.IGNORECASE):
            start, end = int(match.group(1)), int(match.group(2))
            if start <= end and start >= 1:
                teeth.extend(range(start, end + 1))
        
        # 3) "number X, Y, Z" 패턴 (콤마로 구분된 복수)
        multi_pattern = r'\bnumber\s+(\d+(?:\s*,\s*\d+)+)'
        for match in re.finditer(multi_pattern, text, flags=re.IGNORECASE):
            numbers_str = match.group(1)
            numbers = re.findall(r'\d+', numbers_str)
            for n in numbers:
                tooth_num = int(n)
                if 1 <= tooth_num <= 32 or 11 <= tooth_num <= 48:
                    teeth.append(tooth_num)
        
        return teeth


class CommandKeywordComponent(RewardComponent):
    """
    명령어 키워드 정확 매칭 (Label 기반)
    가중치: 0.25
    """
    
    def __init__(self, weight: float = 0.25):
        super().__init__(name="command_keyword", weight=weight)
        
        # 핵심 명령어 정의
        self.critical_commands = {
            'probing', 'pocket', 'depth',
            'mobility', 'furcation', 'bleeding',
            'implant', 'fixture', 'crown', 'bridge', 'pontic',
            'suppuration', 'calculus', 'plaque'
        }
    
    def calculate(self, completion: str, ground_truth: str = None, **kwargs) -> float:
        """
        Label의 각 명령어 단위(;로 구분)를 정확히 매칭
        
        Examples:
            GT: "number 7, pocket depth 5 3 2; number 7, mobility 2"
            
            Gen1: "number 7, pocket depth 5 3 2; number 7, mobility 2"
                → 100% 매칭 → 1.0
            
            Gen2: "number 7, probing 5 3 2; number 7, mobility 2"
                → probing=pocket depth (동의어) → 0.95
            
            Gen3: "number 7, pocket depth 5 3 2"
                → mobility 누락 → 0.5
            
            Gen4: "number 7, bleeding"
                → 완전히 다른 명령어 → 0.1
        """
        # Label을 명령어 단위로 파싱
        gen_commands = self._parse_commands(completion)
        gt_commands = self._parse_commands(ground_truth)
        
        if not gt_commands:
            return 1.0 if not gen_commands else 0.5
        
        if not gen_commands:
            return 0.0
        
        # 각 GT 명령어에 대해 가장 유사한 Gen 명령어 찾기
        match_scores = []
        for gt_cmd in gt_commands:
            best_score = 0.0
            for gen_cmd in gen_commands:
                score = self._command_similarity(gen_cmd, gt_cmd)
                best_score = max(best_score, score)
            match_scores.append(best_score)
        
        # 평균 매칭 점수
        avg_score = sum(match_scores) / len(match_scores)
        
        # 핵심 명령어 누락 페널티
        gt_critical = self._extract_critical_commands(ground_truth)
        gen_critical = self._extract_critical_commands(completion)
        
        if gt_critical:
            critical_recall = len(gen_critical & gt_critical) / len(gt_critical)
            if critical_recall < 0.5:
                avg_score *= 0.7  # 핵심 명령어 절반 이상 누락 시 감점
        
        return avg_score
    
    def _parse_commands(self, text: str) -> List[Dict[str, str]]:
        """
        명령어를 구조화된 딕셔너리로 파싱
        
        Example:
            "number 7, pocket depth 5 3 2" 
            → {"tooth": "7", "command": "pocket depth", "values": "5 3 2"}
        """
        commands = []
        for cmd_str in text.split(';'):
            cmd_str = cmd_str.strip()
            if not cmd_str:
                continue
            
            # "number X, command values" 패턴
            match = re.match(
                r'number\s+([\d\-,\s]+),\s*([^,]+?)(?:,\s*(.+))?$',
                cmd_str,
                re.IGNORECASE
            )
            
            if match:
                tooth = match.group(1).strip()
                command = match.group(2).strip().lower()
                values = match.group(3).strip() if match.group(3) else ""
                
                commands.append({
                    "tooth": tooth,
                    "command": command,
                    "values": values
                })
        
        return commands
    
    def _command_similarity(self, cmd1: Dict, cmd2: Dict) -> float:
        """두 명령어의 유사도"""
        # 치아 번호 일치 필수
        if cmd1["tooth"] != cmd2["tooth"]:
            return 0.0
        
        # 명령어 키워드 비교 (동의어 고려)
        cmd_score = self._keyword_match(cmd1["command"], cmd2["command"])
        
        # 수치값 비교
        if cmd1["values"] and cmd2["values"]:
            value_score = 1.0 if cmd1["values"] == cmd2["values"] else 0.7
        else:
            value_score = 1.0 if not cmd1["values"] and not cmd2["values"] else 0.8
        
        return (cmd_score * 0.7 + value_score * 0.3)
    
    def _keyword_match(self, kw1: str, kw2: str) -> float:
        """키워드 매칭 (동의어 고려)"""
        # 동의어 매핑
        synonyms = {
            'probing': 'pocket_depth',
            'pocket depth': 'pocket_depth',
            'depth': 'pocket_depth',
            'fixture': 'implant',  # 일부 컨텍스트에서
        }
        
        kw1_normalized = synonyms.get(kw1, kw1)
        kw2_normalized = synonyms.get(kw2, kw2)
        
        if kw1_normalized == kw2_normalized:
            return 1.0
        
        # 부분 매칭
        words1 = set(kw1.split())
        words2 = set(kw2.split())
        
        if words1 & words2:
            return 0.7
        
        return 0.0
    
    def _extract_critical_commands(self, text: str) -> Set[str]:
        """핵심 명령어 추출"""
        text_lower = text.lower()
        found = set()
        for cmd in self.critical_commands:
            if cmd in text_lower:
                found.add(cmd)
        return found

class StructuralComponent(RewardComponent):
    """구조 정확도 (가중치: 0.15)"""
    
    def __init__(self, weight: float = 0.15):
        super().__init__(name="command_structural", weight=weight)
    
    def calculate(self, completion: str, ground_truth: str = None, **kwargs) -> float:
        """명령어 개수 및 세미콜론 구조 매칭"""
        gen_count = completion.count(';') + 1
        gt_count = ground_truth.count(';') + 1
        
        count_diff = abs(gen_count - gt_count)
        max_count = max(gen_count, gt_count)
        
        return 1.0 - (count_diff / max_count) if max_count > 0 else 1.0

class NumericalValueComponent(RewardComponent):
    """수치값 정확도 (가중치: 0.10)"""
    
    def __init__(self, weight: float = 0.15):
        super().__init__(name="numerical_value", weight=weight)       

    def calculate(self, completion: str, ground_truth: str = None, **kwargs) -> float:
        """Probing depths, grades 등 숫자 시퀀스 매칭"""
        # "5 3 2" 같은 연속 숫자 패턴 추출
        gen_seqs = re.findall(r'\b(\d+(?:\s+\d+){2,})\b', completion)
        gt_seqs = re.findall(r'\b(\d+(?:\s+\d+){2,})\b', ground_truth)
        
        if not gt_seqs:
            return 1.0
        
        matches = sum(1 for g in gt_seqs if g in gen_seqs)
        return matches / len(gt_seqs)

class CommandRewardFunction(MultiRewardFunction):
    """치과 명령어 GRPO Reward (강화 버전)"""
    
    def __init__(self, config: Dict = None):
        components = [
            ToothNumberComponent(weight=0.50),      # 치아 번호 - 최우선
            CommandKeywordComponent(weight=0.25),   # 명령어 키워드
            StructuralComponent(weight=0.15),       # 구조
            NumericalValueComponent(weight=0.10)    # 수치값
        ]
        
        super().__init__(components=components, config=config)
    
    def reward_func(
        self,
        completions: List[str], 
        ground_truth: List[str] = None,  # Label 직접 전달!
        **kwargs) -> List[float]:
        """
        GRPO용 reward 계산
        
        Args:
            completions: 생성된 명령어들 (K개)
            labels: 정답 label들 (K개, 같은 input에 대해 동일)
        
        Returns:
            rewards: 각 completion의 reward (0~1)
        """
        if not ground_truth:
            raise ValueError("Labels must be provided for dental reward function")
        
        rewards = []
        for completion, label in zip(completions, ground_truth):
            total_reward = 0.0
            
            # 각 component의 reward 계산
            for component in self.components:
                comp_reward = component.calculate(
                    completion=completion,
                    ground_truth=label,
                    **kwargs
                )
                total_reward += comp_reward * component.weight
            
            rewards.append(total_reward)
        
        return rewards
