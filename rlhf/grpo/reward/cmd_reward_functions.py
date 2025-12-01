"""
치과 명령어 GRPO Reward Functions

CommandRewardFunction과 각 component들을 정의합니다.
"""
import logging
import re
import difflib
from typing import Dict, Any, List, Set
from reward.reward_functions import RewardComponent

logger = logging.getLogger(__name__)


class ToothNumberComponent(RewardComponent):
    """
    치아 번호 매칭 (F1 + 엄격한 페널티)
    가중치: 0.40
    """
    
    def __init__(self, weight: float = 0.40):
        super().__init__(name="tooth_number", weight=weight)
        
        # 엄격한 설정
        self.perfect_match_bonus = 0.2  # 완벽 매칭 시 보너스
        self.hallucination_penalty = 0.4  # 없는 치아 추가 시 감점
        self.missing_penalty = 0.3  # 치아 누락 시 감점
        self.wrong_tooth_penalty = 0.5  # 완전히 다른 치아 시 감점
    
    def calculate(self, completion: str, ground_truth: str = None, **kwargs) -> float:
        """
        엄격한 치아 번호 평가
        
        Args:
            completion: 생성된 텍스트 (str)
            ground_truth: 정답 라벨 (str)
            **kwargs: 추가 인자 (labels 등)
        """
        # 형식 안전 처리
        if ground_truth is None and 'labels' in kwargs:
            ground_truth = kwargs.get('labels', [])
            if isinstance(ground_truth, list) and len(ground_truth) > 0:
                ground_truth = ground_truth[0]
        
        if ground_truth is None:
            ground_truth = ""
        
        # str로 변환
        completion = str(completion) if not isinstance(completion, str) else completion
        ground_truth = str(ground_truth) if not isinstance(ground_truth, str) else ground_truth
        
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
    가중치: 0.20
    """
    
    def __init__(self, weight: float = 0.20):
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
        """
        # kwargs에서 labels 추출
        if ground_truth is None and 'labels' in kwargs:
            ground_truth = kwargs.get('labels', [])
            if isinstance(ground_truth, list) and len(ground_truth) > 0:
                ground_truth = ground_truth[0]
        
        # ground_truth 형식 변환
        if isinstance(ground_truth, list):
            if len(ground_truth) > 0:
                ground_truth = ground_truth[0] if isinstance(ground_truth[0], str) else str(ground_truth[0])
            else:
                ground_truth = ""
        elif isinstance(ground_truth, dict):
            ground_truth = ground_truth.get("content", "")
        elif ground_truth is None:
            ground_truth = ""
        
        # completion 형식 변환
        if not isinstance(completion, str):
            if isinstance(completion, list):
                completion = " ".join(str(item) for item in completion)
            else:
                completion = str(completion)
        
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


class InstructionComplianceComponent(RewardComponent):
    """
    Instruction 준수 여부 및 환각 체크 (Reflection & Penalty)
    가중치: 0.20
    """
    def __init__(self, weight: float = 0.20):
        super().__init__(name="instruction_compliance", weight=weight)
        # 금지된 메타 명령어 패턴 (확장되지 않은 형태)
        self.forbidden_patterns = [
            r'\brepeat\b', r'\bothers\b', r'\ball\b', r'\bagain\b', r'\bexcept\b'
        ]
        # 기본 허용 키워드 (이 외의 단어가 GT에 없으면 환각 의심)
        self.base_keywords = {
            'number', 'buccal', 'lingual', 'palatal', 'mesial', 'middle', 'distal',
            'probing', 'pocket', 'depth', 'bleeding', 'suppuration', 'plaque', 'calculus',
            'mobility', 'furcation', 'recession', 'gingival', 'margin',
            'implant', 'fixture', 'crown', 'bridge', 'pontic', 'missing', 'impacted',
            'jump', 'back', 'clear', 'delete'
        }

    def calculate(self, completion: str, ground_truth: str = None, **kwargs) -> float:
        score = 1.0
        completion_lower = str(completion).lower()
        ground_truth_lower = str(ground_truth).lower() if ground_truth else ""

        # 1. 금지된 명령어 사용 체크 (Penalty)
        for pattern in self.forbidden_patterns:
            if re.search(pattern, completion_lower):
                score -= 0.2  # 금지어 사용 시 감점

        # 2. 포맷 규칙 체크: "number N"으로 시작 여부 (Semicolon 단위)
        commands = completion_lower.split(';')
        valid_format_count = 0
        total_commands = 0
        for cmd in commands:
            cmd = cmd.strip()
            if not cmd: continue
            total_commands += 1
            # "number 숫자"로 시작해야 함
            if re.match(r'^number\s+\d+', cmd):
                valid_format_count += 1
        
        if total_commands > 0:
            format_ratio = valid_format_count / total_commands
            # 포맷 준수율이 낮으면 점수 깎음
            score *= format_ratio

        # 3. 환각 체크 (Reflection with GT)
        # GT에 없는 단어를 사용했는지 체크 (단, base_keywords는 제외)
        # 이는 모델이 없는 명령어를 지어내는 것을 방지함
        if ground_truth_lower:
            gen_words = set(re.findall(r'[a-z]+', completion_lower))
            gt_words = set(re.findall(r'[a-z]+', ground_truth_lower))
            
            # GT에 없고, 기본 키워드도 아닌 단어들
            hallucinated_words = gen_words - gt_words - self.base_keywords
            if hallucinated_words:
                # 환각 단어가 있으면 감점 (단어 개수에 비례하여)
                penalty = min(0.5, len(hallucinated_words) * 0.1)
                score -= penalty

        return max(0.0, score)


class SequenceSimilarityComponent(RewardComponent):
    """
    정답과의 시퀀스 유사도 (LCS ratio)
    가중치: 0.10
    """
    def __init__(self, weight: float = 0.10):
        super().__init__(name="sequence_similarity", weight=weight)

    def calculate(self, completion: str, ground_truth: str = None, **kwargs) -> float:
        if not ground_truth:
            return 0.0
        
        completion = str(completion)
        ground_truth = str(ground_truth)
        
        # SequenceMatcher를 이용한 유사도 계산
        matcher = difflib.SequenceMatcher(None, completion, ground_truth)
        return matcher.ratio()


class StructuralComponent(RewardComponent):
    """구조 정확도 (가중치: 0.05)"""
    
    def __init__(self, weight: float = 0.05):
        super().__init__(name="command_structural", weight=weight)
    
    def calculate(self, completion: str, ground_truth: str = None, **kwargs) -> float:
        """명령어 개수 및 세미콜론 구조 매칭"""
        # 형식 안전 처리
        if ground_truth is None:
            return 0.0
        
        completion = str(completion)
        ground_truth = str(ground_truth)
        
        gen_count = completion.count(';') + 1
        gt_count = ground_truth.count(';') + 1
        
        count_diff = abs(gen_count - gt_count)
        max_count = max(gen_count, gt_count)
        
        return 1.0 - (count_diff / max_count) if max_count > 0 else 1.0


class NumericalValueComponent(RewardComponent):
    """수치값 정확도 (가중치: 0.05)"""
    
    def __init__(self, weight: float = 0.05):
        super().__init__(name="numerical_value", weight=weight)       

    def calculate(self, completion: str, ground_truth: str = None, **kwargs) -> float:
        """Probing depths, grades 등 숫자 시퀀스 매칭"""
        # 형식 안전 처리
        if ground_truth is None:
            return 0.0
        
        completion = str(completion)
        ground_truth = str(ground_truth)
        
        # "5 3 2" 같은 연속 숫자 패턴 추출
        try:
            gen_seqs = re.findall(r'\b(\d+(?:\s+\d+){2,})\b', completion)
            gt_seqs = re.findall(r'\b(\d+(?:\s+\d+){2,})\b', ground_truth)
        except Exception as e:
            logger.debug(f"Error in numerical value matching: {e}")
            return 0.0
        
        if not gt_seqs:
            return 1.0
        
        matches = sum(1 for g in gt_seqs if g in gen_seqs)
        return matches / len(gt_seqs)


class ComponentRewardWrapper:
    """개별 component를 독립적인 reward function으로 동작하게 만드는 wrapper"""
    
    __name__ = "ComponentRewardWrapper"
    
    def __init__(self, component, component_name: str, weight: float = 1.0):
        self.component = component
        self.component_name = component_name
        self._name = component_name
    
    def __call__(self, completions, **kwargs) -> List[float]:
        """Component의 reward 계산"""
        rewards = []
        for completion in completions:
            # completion 처리: list, dict, str 모두 처리
            if isinstance(completion, list):
                # 리스트인 경우 첫 번째 항목이나 content 추출
                if len(completion) > 0 and isinstance(completion[0], dict):
                    completion_text = completion[0].get("content", "")
                else:
                    completion_text = " ".join(str(item) for item in completion)
            elif isinstance(completion, dict):
                # dict인 경우 content 추출
                completion_text = completion.get("content", "")
            else:
                # str인 경우 그대로 사용
                completion_text = str(completion)
            
            try:
                reward = self.component.calculate(completion_text, **kwargs)
            except Exception as e:
                logger.warning(f"Error calculating reward for component {self.component_name}: {e}")
                reward = 0.0
            
            rewards.append(reward)
        return rewards
    
    def reward_func(self, completions, **kwargs) -> List[float]:
        """Component의 reward 계산 (별칭)"""
        return self.__call__(completions, **kwargs)
    
    def __repr__(self):
        return f"ComponentRewardWrapper({self.component_name})"
    
    def __str__(self):
        return self.component_name


class CommandRewardFunction:
    """치과 명령어 GRPO Reward"""
    
    def __init__(self, config: Dict = None):
        self.components = [
            ToothNumberComponent(weight=0.40),          # 치아 번호 정확도 (가장 중요)
            CommandKeywordComponent(weight=0.20),       # 명령어 키워드 매칭
            InstructionComplianceComponent(weight=0.20), # 지시 준수 및 환각 체크 (New)
            SequenceSimilarityComponent(weight=0.10),   # 전체 시퀀스 유사도 (New)
            StructuralComponent(weight=0.05),           # 구조적 정확도
            NumericalValueComponent(weight=0.05)        # 수치값 정확도
        ]
        self.config = config or {}
    
    def expand_to_individual_rewards(self) -> List[ComponentRewardWrapper]:
        """CommandRewardFunction을 개별 component reward function들로 확장"""
        individual_rewards = []
        for component in self.components:
            wrapper = ComponentRewardWrapper(
                component=component,
                component_name=f"CommandReward_{component.name}",
                weight=component.weight
            )
            individual_rewards.append(wrapper)
        return individual_rewards
