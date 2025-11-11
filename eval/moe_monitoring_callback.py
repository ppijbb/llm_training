import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict, deque
from typing import Dict, Any, Optional, Callable
import json
import time
import os
from transformers.image_utils import load_image

# GramSpec 분석 도구 import
try:
    from eval.gramspec_moe_analysis import GramSpecAnalyzer
    GRAMSPEC_ANALYSIS_AVAILABLE = True
except ImportError:
    GRAMSPEC_ANALYSIS_AVAILABLE = False

# GramSpec 실제 검증 도구 import
try:
    from eval.gramspec_semantic_validation import GramSpecSemanticValidator
    GRAMSPEC_VALIDATION_AVAILABLE = True
except ImportError:
    GRAMSPEC_VALIDATION_AVAILABLE = False

def _is_main_process() -> bool:
    """Best-effort check for rank-0 to gate logging/plotting on distributed runs."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    
    # 환경변수로도 체크 (DeepSpeed 등에서 사용)
    try:
        rank = int(os.getenv("RANK", "0"))
        return rank == 0
    except (ValueError, TypeError):
        pass
    
    return True

class TorchMoECallback:
    """Pure PyTorch MoE monitoring callback with generation logging"""

    def __init__(
        self,
        num_experts: int,
        log_every_n_steps: int = 1,  # 기본값을 1로 변경하여 매 step마다 로깅
        log_heatmap_every: int = 1000,
        alert_threshold_imbalance: float = 5.0,
        unused_expert_threshold: float = 0.3,
        entropy_threshold: float = 0.1,
        window_size: int = 1000,
        logger: Optional[Any] = None,
        log_to_console: bool = False,
        save_detailed_logs: bool = False,
        log_dir: str = "./moe_logs",
        debug_logging: bool = False,
        enable_generation_logging: bool = True,
        generation_log_dir: str = "./moe_generation_logs",
        max_generation_samples: int = 3,
        generation_log_every: int = 20
    ):
        self.log_every_n_steps = log_every_n_steps
        self.log_heatmap_every = log_heatmap_every
        self.alert_threshold_imbalance = alert_threshold_imbalance
        self.unused_expert_threshold = unused_expert_threshold
        self.num_experts = num_experts
        self.entropy_threshold = entropy_threshold
        self.window_size = window_size
        self.logger = logger
        self.log_to_console = log_to_console
        self.save_detailed_logs = save_detailed_logs
        self.log_dir = log_dir
        self.debug_logging = debug_logging
        self.is_main_process = _is_main_process()

        # Generation logging 설정
        self.enable_generation_logging = enable_generation_logging
        self.generation_log_dir = generation_log_dir
        self.max_generation_samples = max_generation_samples
        self.generation_log_every = generation_log_every
        self.generation_step_count = 0

        # 모델과 토크나이저 (나중에 설정)
        self.model = None
        self.tokenizer = None

        # 내부 상태 (step 제거)
        self.expert_usage_history = defaultdict(lambda: deque(maxlen=window_size))
        self.routing_stats = defaultdict(list)
        self.alerts_history = []
        self.detailed_logs = []

        # hooks 저장소
        self.hooks = []
        self.layer_outputs = {}
        
        # Layer별 expert usage tracking (실제 검증용)
        self.layer_expert_usage_counts = {}  # layer_name -> torch.Tensor [num_experts]
        
        # Wandb step 추적 (monotonically increasing 보장)
        self.last_logged_step = -1
        
        # Vision 모듈 모니터링 (vision_tower, multi_modal_projector)
        self.vision_hooks = []
        self.vision_tower_outputs = []  # vision_tower 출력 히스토리
        self.projector_outputs = []  # projector 출력 히스토리
        self.vision_usage_stats = {
            'vision_tower_calls': 0,
            'projector_calls': 0,
            'pixel_values_received': 0,
            'image_features_generated': 0,
        }
        
        # GramSpec 분석기 (옵션)
        self.gramspec_analyzer = None
        if GRAMSPEC_ANALYSIS_AVAILABLE:
            try:
                self.gramspec_analyzer = GramSpecAnalyzer(num_experts=num_experts, router_dim=128)
            except Exception as e:
                if log_to_console:
                    print(f"Warning: Could not initialize GramSpecAnalyzer: {e}")
        
        # GramSpec 실제 검증기 (옵션)
        self.gramspec_validator = None
        if GRAMSPEC_VALIDATION_AVAILABLE:
            try:
                # num_layers는 register_model에서 설정
                self.gramspec_validator = None  # 나중에 초기화
            except Exception as e:
                if log_to_console:
                    print(f"Warning: Could not initialize GramSpecSemanticValidator: {e}")

        if save_detailed_logs:
            import os
            os.makedirs(log_dir, exist_ok=True)

        if enable_generation_logging:
            import os
            os.makedirs(generation_log_dir, exist_ok=True)
    
    def _log_debug(self, message: str):
        """내부 디버그 메시지 로깅"""
        if self.debug_logging and self.log_to_console:
            print(f"[MoE Debug] {message}")
    
    def register_model(self, model: torch.nn.Module, tokenizer=None):
        """모델에 hooks 등록하고 토크나이저 설정"""
        self.model = model
        self.tokenizer = tokenizer
        self._register_hooks()
        
        # Layer 개수 추출 및 validator 초기화
        if GRAMSPEC_VALIDATION_AVAILABLE:
            try:
                num_layers = self._count_moe_layers(model)
                if num_layers > 0:
                    self.gramspec_validator = GramSpecSemanticValidator(
                        num_layers=num_layers,
                        num_experts=self.num_experts
                    )
                    if self.log_to_console:
                        self._log_debug(f"GramSpecSemanticValidator initialized with {num_layers} layers")
            except Exception as e:
                if self.log_to_console:
                    self._log_debug(f"Warning: Could not initialize validator: {e}")

        if self.enable_generation_logging and tokenizer is None:
            self._log_debug("Warning: Generation logging enabled but no tokenizer provided")

        return self
    
    def _count_moe_layers(self, model: torch.nn.Module) -> int:
        """모델에서 MoE layer 개수 세기"""
        count = 0
        for name, module in model.named_modules():
            if self._is_moe_layer(module):
                count += 1
        return count

    def set_tokenizer(self, tokenizer):
        """토크나이저 설정"""
        self.tokenizer = tokenizer
        return self
    
    def _register_hooks(self):
        """MoE 레이어에 forward hooks 등록"""
        for name, module in self.model.named_modules():
            if self._is_moe_layer(module):
                hook = module.register_forward_hook(
                    self._create_hook_fn(name)
                )
                self.hooks.append(hook)
        
        # Vision 모듈 hooks 등록
        self._register_vision_hooks()
    
    def _is_moe_layer(self, module):
        """MoE 레이어 감지"""
        # 실제 사용 중인 MoE 레이어 클래스들
        moe_class_names = [
            'G3MoESharedExpertsLayer', 'G3MoESparseGRINBlock', 'G3MoEGRINMoE',
            'GRINMoESparseMoeBlock', 'G2MoEGRINMoeLayer',
            # 일반적인 패턴들도 유지
            'gate', 'router', 'expert', 'moe',
            'SparseMLP', 'MixtralSparseMoeBlock', 'SwitchTransformerMLP'
        ]
        
        module_name = module.__class__.__name__
        is_moe = (any(cls_name in module_name for cls_name in moe_class_names) or 
                  hasattr(module, 'gate') or hasattr(module, 'router') or
                  hasattr(module, 'experts'))
        
        # 디버깅 정보
        if is_moe:
            self._log_debug(f"Detected MoE layer: {module_name}")
            
        return is_moe
    
    def _register_vision_hooks(self):
        """Vision tower와 projector에 forward hooks 등록"""
        if self.model is None:
            return
        
        # Vision tower 찾기
        vision_tower = None
        projector = None
        
        # G3MoE 모델 구조에 맞춰 vision_tower와 multi_modal_projector 찾기
        if hasattr(self.model, 'vision_tower'):
            vision_tower = self.model.vision_tower
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'vision_tower'):
            vision_tower = self.model.model.vision_tower
        
        if hasattr(self.model, 'multi_modal_projector'):
            projector = self.model.multi_modal_projector
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'multi_modal_projector'):
            projector = self.model.model.multi_modal_projector
        
        # Vision tower hook 등록
        if vision_tower is not None:
            def vision_tower_hook(module, input, output):
                if not self.is_main_process:
                    return
                try:
                    self.vision_usage_stats['vision_tower_calls'] += 1
                    
                    # Input에서 pixel_values 추출
                    pixel_values = None
                    if isinstance(input, tuple):
                        # 첫 번째 인자가 pixel_values일 수 있음
                        if len(input) > 0 and torch.is_tensor(input[0]):
                            # shape 확인: (batch, channels, height, width)
                            if len(input[0].shape) == 4:
                                pixel_values = input[0]
                    elif isinstance(input, dict):
                        pixel_values = input.get('pixel_values')
                    
                    if pixel_values is not None and torch.is_tensor(pixel_values):
                        batch_size = pixel_values.shape[0] if pixel_values.dim() >= 1 else 1
                        self.vision_usage_stats['pixel_values_received'] += batch_size
                    
                    # Output 통계 수집
                    hidden_state = None
                    if hasattr(output, 'last_hidden_state'):
                        hidden_state = output.last_hidden_state
                    elif isinstance(output, torch.Tensor):
                        hidden_state = output
                    elif isinstance(output, tuple) and len(output) > 0:
                        # BaseModelOutputWithPast 형태일 수 있음
                        if torch.is_tensor(output[0]):
                            hidden_state = output[0]
                    
                    if hidden_state is not None and torch.is_tensor(hidden_state):
                        # 통계 정보만 저장 (메모리 절약)
                        with torch.no_grad():
                            stats = {
                                'shape': list(hidden_state.shape),
                                'mean': hidden_state.float().mean().item() if hidden_state.numel() > 0 else 0.0,
                                'std': hidden_state.float().std().item() if hidden_state.numel() > 0 else 0.0,
                                'min': hidden_state.float().min().item() if hidden_state.numel() > 0 else 0.0,
                                'max': hidden_state.float().max().item() if hidden_state.numel() > 0 else 0.0,
                            }
                            self.vision_tower_outputs.append(stats)
                            # 최근 100개만 유지
                            if len(self.vision_tower_outputs) > 100:
                                self.vision_tower_outputs.pop(0)
                except Exception as e:
                    self._log_debug(f"Error in vision_tower hook: {e}")
            
            hook = vision_tower.register_forward_hook(vision_tower_hook)
            self.vision_hooks.append(hook)
            self._log_debug("Registered vision_tower hook")
        
        # Projector hook 등록
        if projector is not None:
            def projector_hook(module, input, output):
                if not self.is_main_process:
                    return
                try:
                    self.vision_usage_stats['projector_calls'] += 1
                    if isinstance(output, torch.Tensor):
                        batch_size = output.shape[0] if output.dim() >= 1 else 1
                        self.vision_usage_stats['image_features_generated'] += batch_size
                        
                        # 통계 정보만 저장
                        with torch.no_grad():
                            stats = {
                                'shape': list(output.shape),
                                'mean': output.float().mean().item() if output.numel() > 0 else 0.0,
                                'std': output.float().std().item() if output.numel() > 0 else 0.0,
                                'min': output.float().min().item() if output.numel() > 0 else 0.0,
                                'max': output.float().max().item() if output.numel() > 0 else 0.0,
                            }
                            self.projector_outputs.append(stats)
                            # 최근 100개만 유지
                            if len(self.projector_outputs) > 100:
                                self.projector_outputs.pop(0)
                except Exception as e:
                    self._log_debug(f"Error in projector hook: {e}")
            
            hook = projector.register_forward_hook(projector_hook)
            self.vision_hooks.append(hook)
            self._log_debug("Registered multi_modal_projector hook")
    
    def _create_hook_fn(self, layer_name):
        """특정 레이어용 hook 함수 생성"""
        def hook_fn(module, input, output):
            try:
                routing_info = self._extract_routing_info(module, input, output)
                if routing_info:
                    # Store only lightweight, CPU-detached summaries to avoid GPU memory growth
                    lightweight_entry = {}
                    if 'expert_assignments' in routing_info and routing_info['expert_assignments'] is not None:
                        ea = routing_info['expert_assignments']
                        if torch.is_tensor(ea):
                            ea = ea.detach().to('cpu', non_blocking=True)
                        lightweight_entry['expert_assignments'] = ea
                    # Keep num_experts metadata if present
                    if 'num_experts' in routing_info:
                        lightweight_entry['num_experts'] = routing_info['num_experts']
                    # Optionally carry an already-aggregated avg entropy (scalar)
                    if 'avg_routing_entropy' in routing_info and routing_info['avg_routing_entropy'] is not None:
                        val = routing_info['avg_routing_entropy']
                        if torch.is_tensor(val):
                            val = val.detach().to('cpu')
                        lightweight_entry['avg_routing_entropy'] = val
                    if 'ortho_loss' in routing_info and routing_info['ortho_loss'] is not None:
                        val = routing_info['ortho_loss']
                        if torch.is_tensor(val):
                            val = val.detach().to('cpu')
                        lightweight_entry['ortho_loss'] = val
                    if 'aux_loss' in routing_info and routing_info['aux_loss'] is not None:
                        val = routing_info['aux_loss']
                        if torch.is_tensor(val):
                            val = val.detach().to('cpu')
                        lightweight_entry['aux_loss'] = val
                    self.layer_outputs[layer_name] = lightweight_entry
                    # 디버깅 로그는 항상 출력 (step 정보 제거)
                    # self._log_debug(f"{layer_name}: extracted {list(routing_info.keys())}")
                else:
                    self._log_debug(f"{layer_name}: no routing info extracted")
            except Exception as e:
                if self.log_to_console:
                    self._log_debug(f"Warning: Failed to extract routing info from {layer_name}: {e}")
        return hook_fn
    
    @torch.no_grad()
    def _extract_routing_info(self, module, input, output):
        """모듈에서 라우팅 정보 추출"""
        routing_info = {}
        # Lightweight mode: avoid retaining large tensors by default
        lightweight = True

        # G3MoE 모델의 라우팅 정보 추출 (우선순위 1: 모듈에서 저장된 정보)
        if hasattr(module, 'last_selected_experts'):
            selected_experts = module.last_selected_experts
            # selected_experts: [batch*seq, top_k] 형태
            if selected_experts.dim() == 2:
                # top_k experts를 flatten하여 단일 차원으로 변환
                selected_experts_flat = selected_experts.flatten()
                routing_info['expert_assignments'] = selected_experts_flat
                
                # routing_weights도 함께 저장 (entropy 계산용)
                if hasattr(module, 'last_routing_weights'):
                    routing_weights = module.last_routing_weights
                    if routing_weights.dim() == 2:
                        routing_info['routing_probs'] = routing_weights.flatten()
                
                # num_experts 저장
                if hasattr(module, 'last_num_experts'):
                    routing_info['num_experts'] = module.last_num_experts
            else:
                routing_info['expert_assignments'] = selected_experts
        
        # GramSpec 관련 추가 정보 추출
        if hasattr(module, 'router') and hasattr(module.router, 'expression_projector'):
            # Expression projection 정보 (lightweight mode에서는 제외)
            if not lightweight:
                # Expression logits는 너무 클 수 있으므로 평균만 저장
                pass
        
        # Router에서 직접 추출 가능한 정보
        if hasattr(module, 'router'):
            router = module.router
            # Speciality penalty, expression loss 등은 forward 중에 계산되므로
            # 별도로 저장하지 않음 (모니터링 콜백에서는 hook으로 추출 불가)
        
        # 실제 G3MoE/GRIN 모델 구조에 맞춘 추출
        # output이 (hidden_states, router_logits) 튜플인 경우
        elif isinstance(output, tuple) and len(output) == 2:
            hidden_states, router_info_tuple = output
            # G3MoEGRINMoE returns a nested tuple: (hidden_states, (router_logits, ...))
            if isinstance(router_info_tuple, tuple) and len(router_info_tuple) > 0:
                router_logits = router_info_tuple[0]
            else:
                router_logits = router_info_tuple

            if router_logits is not None and torch.is_tensor(router_logits):
                # Compute expert assignments cheaply; skip storing full probs/logits
                expert_assignments = router_logits.argmax(dim=-1)
                routing_info.update({
                    'expert_assignments': expert_assignments,
                })
                if not lightweight:
                    routing_probs = torch.nn.functional.softmax(router_logits, dim=-1)
                    routing_info['routing_probs'] = routing_probs
                    routing_info['gate_logits'] = router_logits
        
        # 다양한 MoE 구현에서 라우팅 정보 추출
        # 속성으로 저장된 경우 (fallback)
        for attr in ['last_expert_assignments', 'expert_assignments', 'selected_experts']:
            if hasattr(module, attr) and 'expert_assignments' not in routing_info:
                routing_info['expert_assignments'] = getattr(module, attr)
                break
        
        for attr in ['last_routing_probs', 'routing_probs', 'gate_probs']:
            if hasattr(module, attr):
                if not lightweight:
                    routing_info['routing_probs'] = getattr(module, attr)
                break
                
        for attr in ['last_gate_logits', 'gate_logits', 'router_logits']:
            if hasattr(module, attr):
                if not lightweight:
                    routing_info['gate_logits'] = getattr(module, attr)
                break
        
        # 2. 기존 output에서 추출 (다른 MoE 구현용)
        if isinstance(output, tuple) and len(output) >= 3:
            # (hidden_states, routing_weights, selected_experts) 형태
            if output[2] is not None:
                routing_info['expert_assignments'] = output[2]
            if not lightweight and output[1] is not None:
                routing_info['routing_probs'] = output[1]
        
        # 3. gate/router 서브모듈에서 추출
        if hasattr(module, 'gate'):
            gate = module.gate
            for attr in ['last_routing_probs', 'routing_probs']:
                if hasattr(gate, attr):
                    if not lightweight:
                        routing_info['routing_probs'] = getattr(gate, attr)
                    break
        if hasattr(module, 'router'):
            router = module.router
            for attr in ['last_routing_probs', 'routing_probs']:
                if hasattr(router, attr):
                    if not lightweight:
                        routing_info['routing_probs'] = getattr(router, attr)
                    break
        # 4. combine_weights 형태로만 제공되는 경우
        cw = getattr(module, 'combine_weights', None)
        if cw is None and isinstance(output, tuple) and len(output) >= 3:
            cw = output[2]
        if cw is not None:
            routing_info['expert_assignments'] = cw.argmax(dim=-1)
            if not lightweight:
                routing_info['routing_probs'] = cw
        
        # num_experts 정보 추출
        if hasattr(module, 'num_experts'):
            routing_info['num_experts'] = module.num_experts
        elif hasattr(module, 'gate') and hasattr(module.gate, 'num_experts'):
            routing_info['num_experts'] = module.gate.num_experts
        elif hasattr(module, 'config') and hasattr(module.config, 'n_routed_experts'):
            routing_info['num_experts'] = module.config.n_routed_experts
        elif hasattr(module, 'config') and hasattr(module.config, 'num_local_experts'):
            routing_info['num_experts'] = module.config.num_local_experts
        elif len(getattr(module, 'experts', [])) > 0:
            routing_info['num_experts'] = len(module.experts)
        
        # Optionally pre-aggregate avg entropy without keeping full probs
        if not lightweight and 'routing_probs' in routing_info and routing_info['routing_probs'] is not None:
            probs = routing_info['routing_probs']
            if probs.dim() > 2:
                probs = probs.view(-1, probs.size(-1))
            token_entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            routing_info['avg_routing_entropy'] = token_entropy.mean()
        
        if hasattr(output, 'ortho_loss'):
            routing_info['ortho_loss'] = output.ortho_loss
        if hasattr(output, 'aux_loss'):
            routing_info['aux_loss'] = output.aux_loss
        
        # GramSpec 관련 메트릭 (router에서 추출 가능한 경우)
        if hasattr(module, 'router'):
            router = module.router
            # Expression loss는 계산 시점에만 존재하므로 직접 추출 불가
            # 대신 router의 expression_projector 상태를 확인
            if hasattr(router, 'expression_projector'):
                # Orthogonal loss는 forward 중에 계산되므로 별도 저장 필요
                # 여기서는 기본 정보만 저장
                pass

        return routing_info if routing_info else None
    
    def on_step_begin(self):
        """Step 시작 시 호출"""
        self.layer_outputs.clear()
        
        # Vision 통계는 누적되므로 초기화하지 않음
        # 대신 step별 사용량을 추적하기 위해 이전 값 저장
        self.prev_vision_stats = self.vision_usage_stats.copy()
    
    def on_step_end(self, current_step: int, **kwargs):
        """Step 종료 시 호출 - current_step은 필수 매개변수"""
        # Main process가 아니면 아무것도 하지 않음
        if not self.is_main_process:
            return
        
        if not self.layer_outputs:
            self._log_debug(f"Step {current_step}: no routing info captured.")
            return
        
        # 메트릭 계산
        step_metrics = self._calculate_step_metrics()
        # wrapper용 last_metrics 저장
        self.last_metrics = step_metrics
        
        # 로깅 (매 step마다 실행)
        if current_step % self.log_every_n_steps == 0:
            self._log_metrics(step_metrics, current_step)
        
        # 히트맵 로깅
        if current_step % self.log_heatmap_every == 0:
            self._log_heatmaps(current_step)
        
        # 경고 체크
        alerts = self._check_alerts(step_metrics)
        if alerts:
            self._handle_alerts(alerts, current_step)
        
        # 상세 로그 저장
        if self.save_detailed_logs:
            self._save_detailed_log(step_metrics, current_step)

        # 생성 로깅 (설정된 주기마다)
        if (self.enable_generation_logging and
            current_step % self.generation_log_every == 0 and
            self.model is not None and
            self.tokenizer is not None):
            self._log_generations(current_step)

    @torch.no_grad()
    def _log_generations(self, current_step: int):
        """모델 생성 결과 로깅"""
        # if not self.is_main_process:
        #     return

        try:
            self.generation_step_count += 1

            sample_image_urls = [
                "https://huggingface.co/spaces/merve/chameleon-7b/resolve/main/bee.jpg",
                "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg",
                "https://ocr.space/Content/Images/table-ocr-original.webp",
            ]

            test_input = self.tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "You are a helpful assistant."}
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image in Korean."},
                            {"type": "image"}
                        ]
                    }
                ],
                # tokenize=True,
                add_generation_prompt=True,
                # return_tensors="pt",
                # return_dict=True,
            )

            generation_logs = []
            sample_count = 0

            # 모델을 evaluation 모드로 전환
            original_mode = self.model.training
            self.model.eval()

            for sample_image_url in sample_image_urls:
                if sample_count >= self.max_generation_samples:
                    break

                try:
                    # 입력 토큰화
                    image = load_image(sample_image_url)

                    inputs = self.tokenizer(
                        text=test_input.replace("<bos>", "")[:-1],
                        images=image,
                        return_tensors="pt")
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                    # 생성 실행
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        num_return_sequences=1,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                    # 생성된 텍스트 디코딩
                    input_length = inputs['input_ids'].shape[1]
                    generated_text = self.tokenizer.decode(
                        outputs[0][input_length:],
                        skip_special_tokens=True
                    )

                    # 로그 데이터 구성
                    log_entry = {
                        "step": current_step,
                        "generation_step": self.generation_step_count,
                        "sample_index": sample_count,
                        "prompt": "Describe this image in Korean.",
                        "generated": generated_text.strip(),
                        "full_response": self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    }

                    generation_logs.append(log_entry)
                    sample_count += 1

                    # 콘솔 로그
                    if self.log_to_console:
                        self._log_debug(f"Generation sample {sample_count}:")
                        self._log_debug(f"  Prompt: {test_input}")
                        self._log_debug(f"  Generated: {generated_text.strip()[:100]}...")

                except Exception as e:
                    self._log_debug(f"Error generating for sample {sample_count}: {e}")
                    sample_count += 1
                    continue

            # 생성 로그 파일 저장
            if generation_logs:
                log_file = os.path.join(
                    self.generation_log_dir,
                    f"generation_log_step_{current_step}_gen_{self.generation_step_count}.json"
                )

                with open(log_file, 'w', encoding='utf-8') as f:
                    json.dump(generation_logs, f, ensure_ascii=False, indent=2)

                self._log_debug(f"Generation logs saved to {log_file}")

                # 로거에 생성 결과 로깅 (Wandb 등)
                # Main process가 아니면 로깅하지 않음
                if self.logger and hasattr(self.logger, 'log') and self.is_main_process:
                    # Step이 감소하지 않도록 보장
                    if current_step > self.last_logged_step:
                        try:
                            gen_log_data = {}
                            for i, log_entry in enumerate(generation_logs):
                                gen_log_data[f'generation/step_{current_step}/sample_{i}/prompt'] = log_entry['prompt']
                                gen_log_data[f'generation/step_{current_step}/sample_{i}/generated'] = log_entry['generated'][:200] + "..."
                            self.logger.log(gen_log_data, step=current_step, commit=True)
                            self.last_logged_step = current_step
                        except Exception as e:
                            self._log_debug(f"Warning: Failed to log generation to wandb at step {current_step}: {e}")

            # 모델을 원래 모드로 복원
            self.model.train(original_mode)

        except Exception as e:
            self._log_debug(f"Error during generation logging: {e}")
            # 모델을 다시 training 모드로 전환
            if self.model is not None:
                self.model.train()


    @torch.no_grad()
    def _calculate_step_metrics(self):
        """현재 step의 메트릭 계산"""
        metrics = {}
        
        for layer_name, routing_info in self.layer_outputs.items():
            layer_metrics = {}
            
            expert_assignments = routing_info.get('expert_assignments')
            routing_probs = routing_info.get('routing_probs')
            
            # ✅ routing_info에서 num_experts 추출 (fallback: self.num_experts)
            num_experts = routing_info.get('num_experts')
            if num_experts is None:
                num_experts = self.num_experts
                self._log_debug(f"Warning: {layer_name} - num_experts not found in routing_info, using fallback: {num_experts}")
            
            # GramSpec 분석 (가능한 경우)
            if self.gramspec_analyzer is not None and hasattr(routing_info, 'gram_matrix'):
                # GramSpec 분석기는 forward hook에서 직접 호출해야 함
                # 여기서는 기본 메트릭만 계산
                pass
            
            if expert_assignments is not None:
                # CPU로 이동 및 clamp
                if torch.is_tensor(expert_assignments):
                    if expert_assignments.is_cuda:
                        expert_assignments = expert_assignments.cpu()
                    expert_assignments = expert_assignments.clamp(0, num_experts - 1)
                
                # Expert 사용 분포
                if expert_assignments.dim() > 1:
                    expert_assignments = expert_assignments.flatten()
                
                # ✅ 올바른 minlength로 bincount 계산
                usage_counts = torch.bincount(expert_assignments.long(), minlength=num_experts)
                self.expert_usage_history[layer_name].append(usage_counts)
                
                # Layer별 expert usage tracking (실제 검증용)
                if layer_name not in self.layer_expert_usage_counts:
                    self.layer_expert_usage_counts[layer_name] = torch.zeros(num_experts, dtype=torch.long)
                self.layer_expert_usage_counts[layer_name] += usage_counts
                
                usage_distribution = usage_counts.float() / (usage_counts.sum() + 1e-8)
                
                # 메트릭 계산
                layer_metrics.update({
                    'usage_counts': usage_counts,
                    'usage_distribution': usage_distribution,
                    'expert_cv': torch.std(usage_distribution) / (torch.mean(usage_distribution) + 1e-8),
                    'max_usage_ratio': usage_distribution.max() / (usage_distribution.mean() + 1e-8),
                    'unused_experts': (usage_counts == 0).sum().item(),
                    'active_experts': (usage_counts > 0).sum().item(),
                })
            
            if routing_probs is not None:
                # 라우팅 엔트로피
                if routing_probs.dim() > 2:
                    routing_probs = routing_probs.view(-1, routing_probs.size(-1))
                
                # 각 토큰의 라우팅 엔트로피
                token_entropy = -torch.sum(routing_probs * torch.log(routing_probs + 1e-10), dim=-1)
                avg_entropy = token_entropy.mean()
                
                layer_metrics.update({
                    'routing_entropy': avg_entropy,
                    'min_entropy': token_entropy.min(),
                    'max_entropy': token_entropy.max(),
                })
            
            metrics[layer_name] = layer_metrics
        
        # Layer-wise balance 분석 (실제 검증 지표)
        if self.gramspec_validator is not None and self.layer_expert_usage_counts:
            # Layer index 추출 (layer_name에서)
            layer_idx_map = {}
            for layer_name in self.layer_expert_usage_counts.keys():
                # layer_name에서 숫자 추출 (예: "model.layers.5.moe" -> 5)
                import re
                match = re.search(r'\.(\d+)\.', layer_name)
                if match:
                    layer_idx = int(match.group(1))
                    layer_idx_map[layer_idx] = self.layer_expert_usage_counts[layer_name]
            
            if layer_idx_map:
                layer_balance_metrics = self.gramspec_validator.analyze_layer_wise_balance(layer_idx_map)
                metrics['_layer_wise_balance'] = layer_balance_metrics
        
        return metrics
    
    def _log_metrics(self, metrics, current_step: int):
        """메트릭 로깅"""
        # Main process가 아니면 로깅하지 않음
        if not self.is_main_process:
            return
        
        # Step이 감소하지 않도록 보장 (엄격한 체크)
        if current_step <= self.last_logged_step:
            self._log_debug(f"Warning: Skipping log at step {current_step} (last logged: {self.last_logged_step})")
            return
        
        log_data = {}
        
        # 레이어별 메트릭
        for layer_name, layer_metrics in metrics.items():
            if layer_name.startswith('_'):
                continue  # 내부 메트릭은 건너뛰기
            for metric_name, value in layer_metrics.items():
                if torch.is_tensor(value) and value.numel() == 1:
                    log_data[f'moe/{layer_name}/{metric_name}'] = value.item()
                elif isinstance(value, (int, float)):
                    log_data[f'moe/{layer_name}/{metric_name}'] = value
        
        # 전체 평균 메트릭
        if metrics:
            # 실제로 값이 있는 경우만 계산 (0으로 fallback하지 않음)
            cv_values = [m['expert_cv'].item() if torch.is_tensor(m['expert_cv']) else m['expert_cv'] 
                         for m in metrics.values() 
                         if isinstance(m, dict) and not isinstance(m, str) and 'expert_cv' in m]
            entropy_values = [m['routing_entropy'].item() if torch.is_tensor(m['routing_entropy']) else m['routing_entropy']
                              for m in metrics.values() 
                              if isinstance(m, dict) and not isinstance(m, str) and 'routing_entropy' in m]
            unused_values = [m['unused_experts'] 
                            for m in metrics.values() 
                            if isinstance(m, dict) and not isinstance(m, str) and 'unused_experts' in m]
            
            if cv_values:
                log_data['moe/avg_expert_cv'] = np.mean(cv_values)
            if entropy_values:
                log_data['moe/avg_routing_entropy'] = np.mean(entropy_values)
            if unused_values:
                log_data['moe/total_unused_experts'] = sum(unused_values)
            
            # Layer-wise balance 메트릭 (실제 검증 지표)
            if '_layer_wise_balance' in metrics:
                balance_metrics = metrics['_layer_wise_balance']
                # 실제로 값이 있는 경우만 로깅 (0으로 fallback하지 않음)
                if 'layer_utilization_cv' in balance_metrics:
                    log_data['validation/layer_utilization_cv'] = balance_metrics['layer_utilization_cv']
                if 'layer_utilization_mean' in balance_metrics:
                    log_data['validation/layer_utilization_mean'] = balance_metrics['layer_utilization_mean']
                if 'layer_entropy_mean' in balance_metrics:
                    log_data['validation/layer_entropy_mean'] = balance_metrics['layer_entropy_mean']
                if 'early_late_utilization_ratio' in balance_metrics:
                    log_data['validation/early_late_ratio'] = balance_metrics['early_late_utilization_ratio']
        
        # Paper 벤치마크 메트릭 추가 (GramSpecAnalyzer가 있는 경우)
        if self.gramspec_analyzer is not None:
            try:
                paper_metrics = self.gramspec_analyzer.get_paper_metrics_summary()
                if paper_metrics:
                    # Load balancing metrics
                    if 'load_balancing' in paper_metrics:
                        lb = paper_metrics['load_balancing']
                        if 'coefficient_of_variation' in lb and lb['coefficient_of_variation'] is not None:
                            log_data['paper/load_balancing/cv'] = lb['coefficient_of_variation']
                        if 'load_imbalance_ratio' in lb and lb['load_imbalance_ratio'] is not None:
                            log_data['paper/load_balancing/imbalance_ratio'] = lb['load_imbalance_ratio']
                        if 'expert_utilization_rate' in lb and lb['expert_utilization_rate'] is not None:
                            log_data['paper/load_balancing/utilization_rate'] = lb['expert_utilization_rate']
                    
                    # Expert specialization metrics
                    if 'expert_specialization' in paper_metrics:
                        es = paper_metrics['expert_specialization']
                        if 'expert_diversity_score' in es and es['expert_diversity_score'] is not None:
                            log_data['paper/expert_specialization/diversity_score'] = es['expert_diversity_score']
                        if 'expert_similarity_mean' in es and es['expert_similarity_mean'] is not None:
                            log_data['paper/expert_specialization/similarity_mean'] = es['expert_similarity_mean']
                        if 'expert_specialization_strength' in es and es['expert_specialization_strength'] is not None:
                            log_data['paper/expert_specialization/specialization_strength'] = es['expert_specialization_strength']
                    
                    # Gram matrix quality
                    if 'gram_matrix_quality' in paper_metrics:
                        gm = paper_metrics['gram_matrix_quality']
                        if 'orthogonality' in gm and gm['orthogonality'] is not None:
                            log_data['paper/gram_matrix/orthogonality'] = gm['orthogonality']
                        if 'orthogonality_std' in gm and gm['orthogonality_std'] is not None:
                            log_data['paper/gram_matrix/orthogonality_std'] = gm['orthogonality_std']
                    
                    # Routing quality
                    if 'routing_quality' in paper_metrics:
                        rq = paper_metrics['routing_quality']
                        if 'routing_confidence' in rq and rq['routing_confidence'] is not None:
                            log_data['paper/routing/confidence'] = rq['routing_confidence']
                        if 'cosine_similarity_mean' in rq and rq['cosine_similarity_mean'] is not None:
                            log_data['paper/routing/cosine_similarity_mean'] = rq['cosine_similarity_mean']
            except Exception as e:
                self._log_debug(f"Warning: Failed to get paper metrics: {e}")
        
        # Vision 모듈 사용 통계 추가 (step별 증가량 계산)
        if hasattr(self, 'prev_vision_stats'):
            step_vision_tower_calls = self.vision_usage_stats['vision_tower_calls'] - self.prev_vision_stats.get('vision_tower_calls', 0)
            step_projector_calls = self.vision_usage_stats['projector_calls'] - self.prev_vision_stats.get('projector_calls', 0)
            step_pixel_values = self.vision_usage_stats['pixel_values_received'] - self.prev_vision_stats.get('pixel_values_received', 0)
            step_image_features = self.vision_usage_stats['image_features_generated'] - self.prev_vision_stats.get('image_features_generated', 0)
            
            if step_vision_tower_calls > 0 or step_projector_calls > 0:
                log_data['vision/vision_tower_calls_per_step'] = step_vision_tower_calls
                log_data['vision/projector_calls_per_step'] = step_projector_calls
                log_data['vision/pixel_values_per_step'] = step_pixel_values
                log_data['vision/image_features_per_step'] = step_image_features
                
                # 누적 통계도 함께 로깅
                log_data['vision/vision_tower_calls_total'] = self.vision_usage_stats['vision_tower_calls']
                log_data['vision/projector_calls_total'] = self.vision_usage_stats['projector_calls']
                log_data['vision/pixel_values_total'] = self.vision_usage_stats['pixel_values_received']
                log_data['vision/image_features_total'] = self.vision_usage_stats['image_features_generated']
                
                # Vision tower 출력 통계
                if self.vision_tower_outputs:
                    recent_outputs = self.vision_tower_outputs[-10:]  # 최근 10개
                    log_data['vision/tower_output_mean'] = np.mean([o['mean'] for o in recent_outputs])
                    log_data['vision/tower_output_std'] = np.mean([o['std'] for o in recent_outputs])
                    log_data['vision/tower_output_min'] = np.min([o['min'] for o in recent_outputs])
                    log_data['vision/tower_output_max'] = np.max([o['max'] for o in recent_outputs])
                
                # Projector 출력 통계
                if self.projector_outputs:
                    recent_outputs = self.projector_outputs[-10:]  # 최근 10개
                    log_data['vision/projector_output_mean'] = np.mean([o['mean'] for o in recent_outputs])
                    log_data['vision/projector_output_std'] = np.mean([o['std'] for o in recent_outputs])
                    log_data['vision/projector_output_min'] = np.min([o['min'] for o in recent_outputs])
                    log_data['vision/projector_output_max'] = np.max([o['max'] for o in recent_outputs])
                
                # Vision 사용률 (이미지가 있는 배치 비율)
                if step_vision_tower_calls > 0:
                    log_data['vision/vision_usage_rate'] = 1.0  # 이 step에서 vision이 사용됨
                else:
                    log_data['vision/vision_usage_rate'] = 0.0  # 이 step에서 vision이 사용되지 않음
        
        # 로거에 전송 (step 명시적으로 전달하여 monotonically increasing 보장)
        # Main process가 아니면 로깅하지 않음 (이미 위에서 체크했지만 이중 체크)
        if self.logger and self.is_main_process:
            if hasattr(self.logger, 'log'):
                # Wandb의 경우: step이 증가하는 경우에만 로깅
                # commit=True는 기본값 (step을 증가시킴)
                if current_step > self.last_logged_step:
                    try:
                        self.logger.log(log_data, step=current_step, commit=True)
                        self.last_logged_step = current_step
                    except Exception as e:
                        self._log_debug(f"Warning: Failed to log to wandb at step {current_step}: {e}")
            elif hasattr(self.logger, 'add_scalars'):  # TensorBoard
                if current_step > self.last_logged_step:
                    for key, value in log_data.items():
                        self.logger.add_scalar(key, value, current_step)
                    self.last_logged_step = current_step
        
        # 콘솔 출력
        if self.log_to_console:
            self._log_debug(f"Step {current_step} MoE Metrics:")
            for key, value in log_data.items():
                if 'avg_' in key or 'total_' in key or 'paper/' in key:
                    if value is not None and isinstance(value, (int, float)):
                        self._log_debug(f"  {key}: {value:.4f}")
                    else:
                        self._log_debug(f"  {key}: {value}")
    
    def _log_heatmaps(self, current_step: int):
        """Expert 사용률 히트맵 로깅"""
        # Main process가 아니면 로깅하지 않음
        if not self.is_main_process:
            return
        
        # Step이 감소하지 않도록 보장
        if current_step <= self.last_logged_step:
            return
        
        if not self.expert_usage_history:
            if self.debug_logging:
                self._log_debug(f"No expert usage history available for heatmap at step {current_step}")
            return
        
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            heatmap_created = False
            
            for layer_name in self.expert_usage_history:
                history = self.expert_usage_history[layer_name]
                # 최소 2개 이상의 데이터 필요 (10개에서 완화)
                if len(history) < 2:
                    if self.debug_logging:
                        self._log_debug(f"Insufficient data for {layer_name} heatmap (need at least 2 steps, got {len(history)})")
                    continue
                
                try:
                    # 모든 텐서를 동일한 크기로 맞추기 위해 최대 크기로 패딩
                    usage_tensors = list(history)
                    if not usage_tensors:
                        continue
                    
                    # 빈 텐서 필터링
                    valid_tensors = [t for t in usage_tensors if t.numel() > 0]
                    if not valid_tensors:
                        continue
                    
                    max_size = max(tensor.size(0) for tensor in valid_tensors)
                    if max_size == 0:
                        continue
                    
                    # 각 텐서를 최대 크기로 패딩
                    padded_tensors = []
                    for tensor in valid_tensors:
                        if tensor.size(0) < max_size:
                            padding = torch.zeros(max_size - tensor.size(0), dtype=tensor.dtype)
                            padded_tensor = torch.cat([tensor, padding])
                        else:
                            padded_tensor = tensor
                        padded_tensors.append(padded_tensor)
                    
                    if not padded_tensors:
                        continue
                    
                    usage_matrix = torch.stack(padded_tensors)
                    usage_matrix = usage_matrix.float()
                    
                    # 정규화
                    row_sums = usage_matrix.sum(dim=1, keepdim=True)
                    usage_matrix = usage_matrix / (row_sums + 1e-8)
                    
                    # 히트맵 생성
                    plt.figure(figsize=(12, 6))
                    sns.heatmap(usage_matrix.T.numpy(), 
                                cmap='YlOrRd', 
                                xticklabels=False,
                                yticklabels=True,
                                cbar_kws={'label': 'Usage Ratio'})
                    plt.title(f'{layer_name} Expert Usage Distribution (Step {current_step})')
                    plt.xlabel('Time Steps')
                    plt.ylabel('Expert Index')
                    plt.tight_layout()
                    
                    # 로거에 전송 (step 명시적으로 전달)
                    # Main process가 아니면 로깅하지 않음
                    if self.logger and hasattr(self.logger, 'log') and self.is_main_process:
                        if current_step > self.last_logged_step:
                            try:
                                import wandb
                                self.logger.log({
                                    f'moe/{layer_name}/usage_heatmap': wandb.Image(plt)
                                }, step=current_step, commit=True)
                                self.last_logged_step = current_step
                            except Exception as e:
                                self._log_debug(f"Warning: Failed to log heatmap to logger: {e}")
                    
                    # 파일로 저장
                    if self.save_detailed_logs:
                        try:
                            safe_layer_name = layer_name.replace(".", "_").replace("/", "_")
                            heatmap_path = os.path.join(self.log_dir, f'{safe_layer_name}_heatmap_step_{current_step}.png')
                            plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
                            if self.debug_logging:
                                self._log_debug(f"Heatmap saved to {heatmap_path}")
                        except Exception as e:
                            self._log_debug(f"Warning: Failed to save heatmap: {e}")
                    
                    plt.close()
                    heatmap_created = True
                    
                except Exception as e:
                    import traceback
                    self._log_debug(f"Error creating heatmap for {layer_name}: {e}\n{traceback.format_exc()}")
                    continue
            
            if not heatmap_created and self.debug_logging:
                self._log_debug(f"No heatmaps were created at step {current_step}")
                
        except ImportError as e:
            self._log_debug(f"Warning: matplotlib/seaborn not available for heatmap logging: {e}")
        except Exception as e:
            import traceback
            self._log_debug(f"Error during heatmap logging: {e}\n{traceback.format_exc()}")
    
    def _check_alerts(self, metrics):
        """경고 상황 체크"""
        alerts = []
        
        for layer_name, layer_metrics in metrics.items():
            # 심각한 불균형
            if 'max_usage_ratio' in layer_metrics:
                ratio = layer_metrics['max_usage_ratio']
                if isinstance(ratio, torch.Tensor):
                    ratio = ratio.item()
                if ratio > self.alert_threshold_imbalance:
                    alerts.append({
                        'type': 'severe_imbalance',
                        'layer': layer_name,
                        'severity': ratio,
                        'message': f'{layer_name}: Severe expert imbalance (ratio: {ratio:.2f})'
                    })
            
            # 사용되지 않는 experts
            if 'unused_experts' in layer_metrics and 'usage_counts' in layer_metrics:
                unused = layer_metrics['unused_experts']
                usage_counts = layer_metrics['usage_counts']
                if torch.is_tensor(usage_counts):
                    total_experts = usage_counts.numel()
                else:
                    total_experts = len(usage_counts) if hasattr(usage_counts, '__len__') else self.num_experts
                
                if total_experts > 0 and unused / total_experts > self.unused_expert_threshold:
                    alerts.append({
                        'type': 'unused_experts',
                        'layer': layer_name,
                        'unused_count': unused,
                        'total_experts': total_experts,
                        'message': f'{layer_name}: {unused}/{total_experts} experts unused'
                    })
            
            # 낮은 라우팅 엔트로피
            if 'routing_entropy' in layer_metrics:
                entropy = layer_metrics['routing_entropy']
                if isinstance(entropy, torch.Tensor):
                    entropy = entropy.item()
                if entropy < self.entropy_threshold:
                    alerts.append({
                        'type': 'low_entropy',
                        'layer': layer_name,
                        'entropy': entropy,
                        'message': f'{layer_name}: Low routing entropy ({entropy:.4f})'
                    })
        
        return alerts
    
    def _handle_alerts(self, alerts, current_step: int):
        """경고 처리"""
        # rank 0에서만 로깅 수행
        if not self.is_main_process:
            return
            
        for alert in alerts:
            self.alerts_history.append({
                'step': current_step,
                'timestamp': time.time(),
                **alert
            })
            
            if self.log_to_console:
                self._log_debug(f"⚠️  MoE Alert at step {current_step}: {alert['message']}")
            
            # Main process가 아니면 로깅하지 않음 (이미 위에서 체크했지만 이중 체크)
            if self.logger and hasattr(self.logger, 'log') and self.is_main_process:
                # Step이 감소하지 않도록 보장
                if current_step > self.last_logged_step:
                    try:
                        self.logger.log({
                            f'alerts/{alert["type"]}': 1,
                            f'alerts/{alert["layer"]}_severity': alert.get('severity', 1)
                        }, step=current_step, commit=True)
                        self.last_logged_step = current_step
                    except Exception as e:
                        self._log_debug(f"Warning: Failed to log alert to wandb at step {current_step}: {e}")
    
    def _save_detailed_log(self, metrics, current_step: int):
        """상세 로그 저장"""
        if not self.is_main_process:
            return
        log_entry = {
            'step': current_step,
            'timestamp': time.time(),
            'metrics': {
                layer: {k: v.tolist() if torch.is_tensor(v) else v 
                       for k, v in layer_metrics.items()}
                for layer, layer_metrics in metrics.items()
            }
        }
        
        with open(f'{self.log_dir}/detailed_log_step_{current_step}.json', 'w') as f:
            json.dump(log_entry, f, indent=2)
    
    def cleanup(self):
        """정리 작업"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
        # Vision hooks 정리
        for hook in self.vision_hooks:
            hook.remove()
        self.vision_hooks.clear()
    
    def get_summary(self):
        """전체 훈련에 대한 요약 통계"""
        summary = {
            'total_alerts': len(self.alerts_history),
            'alert_types': {}
        }
        
        # 경고 유형별 집계
        for alert in self.alerts_history:
            alert_type = alert['type']
            if alert_type not in summary['alert_types']:
                summary['alert_types'][alert_type] = 0
            summary['alert_types'][alert_type] += 1
        
        return summary

from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from typing import Dict, Any

class TransformersMoECallbackWrapper(TrainerCallback):
    """Transformers TrainerCallback wrapper for TorchMoECallback"""
    
    def __init__(self, torch_callback: TorchMoECallback):
        self.torch_callback = torch_callback
        self._model_registered = False
    
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        tokenizer=None,
        **kwargs
    ):
        """훈련 시작 시 모델과 토크나이저 등록"""
        if model is not None and not self._model_registered:
            self.torch_callback.register_model(model, tokenizer)
            self._model_registered = True
            self.torch_callback._log_debug(f"MoE monitoring registered for model with {len(self.torch_callback.hooks)} MoE layers")

            if self.torch_callback.enable_generation_logging:
                if tokenizer is not None:
                    self.torch_callback._log_debug("Generation logging enabled with tokenizer")
                else:
                    self.torch_callback._log_debug("Warning: Generation logging enabled but no tokenizer provided")
    
    def on_step_begin(
        self, 
        args: TrainingArguments, 
        state: TrainerState, 
        control: TrainerControl, 
        **kwargs
    ):
        """Step 시작"""
        self.torch_callback.on_step_begin()
    
    def on_step_end(
        self, 
        args: TrainingArguments, 
        state: TrainerState, 
        control: TrainerControl, 
        logs: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        """Step 종료"""
        # rank 0에서만 MoE 콜백 실행
        if not self.torch_callback.is_main_process:
            return
            
        # PyTorch callback 호출 - Transformers의 global_step 사용
        self.torch_callback.on_step_end(current_step=state.global_step)
        
        # logs에 MoE 메트릭 추가 (선택사항)
        if logs is not None and hasattr(self.torch_callback, 'last_metrics'):
            moe_metrics = {}
            for layer_name, layer_metrics in self.torch_callback.last_metrics.items():
                for metric_name, value in layer_metrics.items():
                    if torch.is_tensor(value) and value.numel() == 1:
                        moe_metrics[f'moe_{layer_name}_{metric_name}'] = value.item()
                    elif isinstance(value, (int,  float)):
                        moe_metrics[f'moe_{layer_name}_{metric_name}'] = value
            
            logs.update(moe_metrics)

        try:
            import deepspeed.accelerator as ds_acc
            ds_acc.get_accelerator().empty_cache()
        except Exception:
            pass
    
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        tokenizer=None,
        eval_dataloader=None,
        **kwargs
    ):
        """Evaluation 시점에 GramSpec 지표 측정"""
        if not self.torch_callback.is_main_process:
            return
        
        # GramSpecAnalyzer가 없으면 스킵
        if self.torch_callback.gramspec_analyzer is None:
            return
        
        try:
            # Evaluation 모드로 전환
            original_training = model.training if model is not None else None
            if model is not None:
                model.eval()
            
            # Analyzer 초기화 (eval 전용)
            eval_analyzer = self.torch_callback.gramspec_analyzer
            eval_analyzer.reset()  # 이전 데이터 초기화
            
            # Router와 MoE Block에서 routing 정보 수집을 위한 hook 등록
            from eval.evaluate_checkpoint_model import RoutingInfoCollector
            collector = RoutingInfoCollector(eval_analyzer)
            collector.register_hooks(model)
            
            # Eval dataloader로 forward pass 실행
            # eval_dataloader가 None이면 trainer에서 가져오기 시도
            dataloader = eval_dataloader
            if dataloader is None and 'trainer' in kwargs:
                trainer = kwargs['trainer']
                if hasattr(trainer, 'get_eval_dataloader'):
                    try:
                        dataloader = trainer.get_eval_dataloader()
                    except:
                        pass
            
            if dataloader is not None:
                self.torch_callback._log_debug(f"Running evaluation metrics collection at step {state.global_step}...")
                
                num_samples = 0
                max_eval_samples = getattr(args, 'max_eval_samples', 100)  # 기본값 100
                
                with torch.no_grad():
                    for batch in dataloader:
                        if num_samples >= max_eval_samples:
                            break
                        
                        # Batch를 device로 이동
                        batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                                for k, v in batch.items()}
                        
                        # Forward pass (routing 정보 수집)
                        try:
                            outputs = model(**batch)
                            # Batch size 계산
                            batch_size = 1
                            if 'input_ids' in batch:
                                batch_size = batch['input_ids'].shape[0]
                            elif 'pixel_values' in batch:
                                batch_size = batch['pixel_values'].shape[0]
                            num_samples += batch_size
                        except Exception as e:
                            self.torch_callback._log_debug(f"Error in eval forward pass: {e}")
                            continue
                
                # 수집된 데이터 분석
                self.torch_callback._log_debug(f"Analyzing {num_samples} eval samples...")
                eval_results = collector.analyze_collected_data(
                    num_experts=self.torch_callback.num_experts,
                    router_dim=128
                )
                
                # Hook 제거
                collector.remove_hooks()
                
                # 결과를 trainer logs에 추가
                if 'aggregated_metrics' in eval_results:
                    eval_metrics = eval_results['aggregated_metrics']
                    
                    # 논문용 지표들을 trainer에 로깅
                    eval_log_data = {}
                    
                    # Load Balancing 지표
                    if 'final_load_balancing_cv' in eval_metrics:
                        eval_log_data['eval/load_balancing/cv'] = eval_metrics['final_load_balancing_cv']
                    if 'final_load_imbalance_ratio' in eval_metrics:
                        eval_log_data['eval/load_balancing/imbalance_ratio'] = eval_metrics['final_load_imbalance_ratio']
                    if 'expert_utilization_rate' in eval_metrics:
                        eval_log_data['eval/load_balancing/utilization_rate'] = eval_metrics['expert_utilization_rate']
                    if 'final_maxvio' in eval_metrics:
                        eval_log_data['eval/load_balancing/maxvio'] = eval_metrics['final_maxvio']
                    if 'final_aux_loss' in eval_metrics:
                        eval_log_data['eval/load_balancing/aux_loss'] = eval_metrics['final_aux_loss']
                    
                    # Expert Specialization 지표
                    if 'final_expert_diversity_score' in eval_metrics:
                        eval_log_data['eval/specialization/diversity_score'] = eval_metrics['final_expert_diversity_score']
                    if 'final_expert_similarity_mean' in eval_metrics:
                        eval_log_data['eval/specialization/similarity_mean'] = eval_metrics['final_expert_similarity_mean']
                    if 'final_expert_specialization_strength' in eval_metrics:
                        eval_log_data['eval/specialization/specialization_strength'] = eval_metrics['final_expert_specialization_strength']
                    
                    # Gram Matrix Quality
                    if 'avg_gram_orthogonality' in eval_metrics:
                        eval_log_data['eval/gram_matrix/orthogonality'] = eval_metrics['avg_gram_orthogonality']
                    
                    # Paper summary도 로깅
                    if 'paper_summary' in eval_results:
                        paper_summary = eval_results['paper_summary']
                        if 'load_balancing' in paper_summary:
                            lb = paper_summary['load_balancing']
                            for key, value in lb.items():
                                if isinstance(value, (int, float)):
                                    eval_log_data[f'eval/paper/load_balancing/{key}'] = value
                    
                    # Logger에 전송 (trainer.log를 통해 전달)
                    if 'trainer' in kwargs:
                        trainer = kwargs['trainer']
                        if hasattr(trainer, 'log'):
                            trainer.log(eval_log_data)
                    
                    # 콘솔 출력
                    if self.torch_callback.log_to_console:
                        self.torch_callback._log_debug(f"\n{'='*60}")
                        self.torch_callback._log_debug(f"Evaluation Metrics (Step {state.global_step}):")
                        self.torch_callback._log_debug(f"{'='*60}")
                        for key, value in eval_log_data.items():
                            if isinstance(value, (int, float)):
                                self.torch_callback._log_debug(f"  {key}: {value:.4f}")
                        self.torch_callback._log_debug(f"{'='*60}\n")
                    
                    # 상세 결과 저장
                    if self.torch_callback.save_detailed_logs:
                        eval_result_file = os.path.join(
                            self.torch_callback.log_dir,
                            f"eval_metrics_step_{state.global_step}.json"
                        )
                        with open(eval_result_file, 'w') as f:
                            json.dump(eval_results, f, indent=2)
                        self.torch_callback._log_debug(f"Eval metrics saved to {eval_result_file}")
            else:
                self.torch_callback._log_debug("⚠️ No eval dataloader available for metrics collection")
            
            # 모델을 원래 모드로 복원
            if model is not None and original_training is not None:
                model.train(original_training)
                
        except Exception as e:
            import traceback
            self.torch_callback._log_debug(f"Error during evaluation metrics collection: {e}")
            self.torch_callback._log_debug(traceback.format_exc())
            # 모델을 원래 모드로 복원
            if model is not None and original_training is not None:
                model.train(original_training)
    
    def on_train_end(
        self, 
        args: TrainingArguments, 
        state: TrainerState, 
        control: TrainerControl, 
        **kwargs
    ):
        """훈련 종료"""
        summary = self.torch_callback.get_summary()
        if self.torch_callback.log_to_console:
            self.torch_callback._log_debug("\n" + "="*50)
            self.torch_callback._log_debug("MoE Training Summary:")
            self.torch_callback._log_debug(f"Total alerts: {summary['total_alerts']}")
            if summary['alert_types']:
                self.torch_callback._log_debug("Alert breakdown:")
                for alert_type, count in summary['alert_types'].items():
                    self.torch_callback._log_debug(f"  {alert_type}: {count}")
            self.torch_callback._log_debug("="*50)
        
        # 정리
        self.torch_callback.cleanup()

def create_moe_callback_for_transformers(
    log_every_n_steps: int = 100,
    logger=None,
    enable_generation_logging: bool = True,
    generation_log_dir: str = "./moe_generation_logs",
    max_generation_samples: int = 3,
    generation_log_every: int = 100,
    **kwargs
) -> TransformersMoECallbackWrapper:
    """Transformers용 MoE 콜백 생성 편의 함수"""

    torch_callback = TorchMoECallback(
        log_every_n_steps=log_every_n_steps,
        logger=logger,
        enable_generation_logging=enable_generation_logging,
        generation_log_dir=generation_log_dir,
        max_generation_samples=max_generation_samples,
        generation_log_every=generation_log_every,
        **kwargs
    )

    return TransformersMoECallbackWrapper(torch_callback)

def create_moe_callback_for_pytorch(
    model: torch.nn.Module,
    log_every_n_steps: int = 100,
    logger=None,
    tokenizer=None,
    enable_generation_logging: bool = True,
    generation_log_dir: str = "./moe_generation_logs",
    max_generation_samples: int = 3,
    generation_log_every: int = 100,
    **kwargs
) -> TorchMoECallback:
    """순수 PyTorch용 MoE 콜백 생성 편의 함수"""

    callback = TorchMoECallback(
        log_every_n_steps=log_every_n_steps,
        logger=logger,
        enable_generation_logging=enable_generation_logging,
        generation_log_dir=generation_log_dir,
        max_generation_samples=max_generation_samples,
        generation_log_every=generation_log_every,
        **kwargs
    )

    return callback.register_model(model, tokenizer)
