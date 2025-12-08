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

# 벤치마크 도구 import
try:
    from eval.analyze_expert_specialization import (
        collect_expert_activations,
        compute_expert_similarity,
        analyze_expert_task_correlation,
    )
    EXPERT_SPECIALIZATION_AVAILABLE = True
except ImportError:
    EXPERT_SPECIALIZATION_AVAILABLE = False

try:
    from eval.run_gramspec_validation import (
        evaluate_model_perplexity,
        run_expression_ablation_study,
        run_information_processing_comparison,
    )
    GRAMSPEC_VALIDATION_SCRIPT_AVAILABLE = True
except ImportError:
    GRAMSPEC_VALIDATION_SCRIPT_AVAILABLE = False

try:
    from eval.measure_efficiency import (
        measure_forward_throughput,
        measure_generation_latency,
        estimate_flops,
    )
    EFFICIENCY_MEASUREMENT_AVAILABLE = True
except ImportError:
    EFFICIENCY_MEASUREMENT_AVAILABLE = False

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

def _get_process_info() -> dict:
    """현재 프로세스 정보 반환 (디버깅용)"""
    info = {
        'rank': None,
        'world_size': None,
        'local_rank': None,
        'RANK': os.getenv("RANK", "N/A"),
        'LOCAL_RANK': os.getenv("LOCAL_RANK", "N/A"),
        'WORLD_SIZE': os.getenv("WORLD_SIZE", "N/A"),
    }
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            info['rank'] = dist.get_rank()
            info['world_size'] = dist.get_world_size()
            try:
                info['local_rank'] = dist.get_rank() % torch.cuda.device_count()
            except:
                pass
    except Exception:
        pass
    return info

class TorchMoECallback:
    """Pure PyTorch MoE monitoring callback with generation logging"""

    def __init__(
        self,
        num_experts: int,
        log_every_n_steps: int = 1,  # 기본값을 1로 변경하여 매 step마다 로깅
        log_heatmap_every: int = 1000,
        log_tsne_every: int = 5000,  # t-SNE 시각화 주기 (계산 비용이 높으므로 기본값을 크게 설정)
        tsne_sample_size: int = 2000,  # t-SNE 계산용 샘플 크기
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
        generation_log_every: int = 20,
        force_all_ranks: bool = True
    ):
        self.log_every_n_steps = log_every_n_steps
        self.log_heatmap_every = log_heatmap_every
        self.log_tsne_every = log_tsne_every
        self.tsne_sample_size = tsne_sample_size
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
        self.force_all_ranks = bool(force_all_ranks)
        self.is_main_process = True  # 항상 모든 프로세스에서 실행
        self.last_logged_step = -1

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

        # Pending 로깅 정보 (on_log에서 사용)
        self.pending_metrics = {}   # step -> log_data
        self.pending_heatmaps = {}  # step -> heatmap_data
        self.pending_alerts = {}    # step -> alert_data

        # hooks 저장소
        self.hooks = []
        self.layer_outputs = {}
        
        # Layer별 expert usage tracking (실제 검증용)
        self.layer_expert_usage_counts = {}  # layer_name -> torch.Tensor [num_experts]
        
        # t-SNE 시각화용 데이터 버퍼 (메모리 절약을 위해 최근 N개 step만 유지)
        self.tsne_data_buffer = defaultdict(lambda: {
            'hidden_states': deque(maxlen=50),  # 최근 50개 step의 hidden states
            'expert_assignments': deque(maxlen=50),
            'routing_weights': deque(maxlen=50)
        })
        
        
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
        self.vision_modules_info = {
            'vision_tower': {'module': None, 'name': None},
            'projector': {'module': None, 'name': None}
        }
        
        # GramSpec 분석기 (옵션)
        self.gramspec_analyzer = None
        if GRAMSPEC_ANALYSIS_AVAILABLE:
            try:
                self.gramspec_analyzer = GramSpecAnalyzer(num_experts=num_experts, router_dim=128)
            except Exception as e:
                self._log_debug(f"Warning: Could not initialize GramSpecAnalyzer: {e}")
        
        # GramSpec 실제 검증기 (옵션)
        self.gramspec_validator = None
        if GRAMSPEC_VALIDATION_AVAILABLE:
            try:
                # num_layers는 register_model에서 설정
                self.gramspec_validator = None  # 나중에 초기화
            except Exception as e:
                self._log_debug(f"Warning: Could not initialize GramSpecSemanticValidator: {e}")

        if save_detailed_logs:
            import os
            os.makedirs(log_dir, exist_ok=True)

        if enable_generation_logging:
            import os
            os.makedirs(generation_log_dir, exist_ok=True)
    
    def _log_debug(self, message: str):
        """내부 디버그 메시지 로깅"""
        # log_to_console이 True일 때만 출력 (debug_logging은 wandb에만 기록)
        if self.log_to_console:
            prefix = "[MoE Debug]" if self.debug_logging else "[MoE]"
            print(f"{prefix} {message}")
    
    def register_model(self, model: torch.nn.Module, tokenizer=None):
        """모델에 hooks 등록하고 토크나이저 설정 (치명적 버그 수정: DeepSpeed 래핑 대응)"""
        # DeepSpeed 래핑 처리 (model.module이 실제 모델)
        actual_model = model.module if hasattr(model, 'module') else model
        self.model = actual_model  # ← 이거 안 하면 hook이 wrapper에 걸림
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
                    self._log_debug(f"GramSpecSemanticValidator initialized with {num_layers} layers")
            except Exception as e:
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
        moe_count = 0
        for name, module in self.model.named_modules():
            if self._is_moe_layer(module):
                hook = module.register_forward_hook(
                    self._create_hook_fn(name)
                )
                self.hooks.append(hook)
                moe_count += 1
        
        if moe_count == 0:
            self._log_debug("❌ WARNING: No MoE layers found! Model structure:")
            # 모델 구조 일부 출력
            for name, module in list(self.model.named_modules())[:20]:
                self._log_debug(f"    - {name}: {type(module).__name__}")
        
        self._log_debug(f"📊 Total MoE hooks registered: {len(self.hooks)}")
        
        # Vision 모듈 hooks 등록
        self._register_vision_hooks()
    
    def _is_moe_layer(self, module):
        """MoE 레이어 감지"""
        # 실제 사용 중인 MoE 레이어 클래스들 (치명적 버그 수정: GramSpecMoEGRINMoE 추가)
        moe_class_names = [
            'GramSpecMoEGRINMoE',      # ← 이거 없으면 hook 0개 (가장 중요!)
            'G3MoESharedExpertsLayer', 
            'G3MoESparseGRINBlock', 
            'G3MoEGRINMoE',
            'GRINMoESparseMoeBlock', 
            'G2MoEGRINMoeLayer', 
            'GramSpecMoEBlock',
            'MixtralSparseMoeBlock',   # 일반적인 패턴들도 유지
            'SparseMLP', 
            'SwitchTransformerMLP'
        ]
        
        module_name = module.__class__.__name__
        
        # 클래스 이름으로 체크
        is_moe_by_name = any(cls_name in module_name for cls_name in moe_class_names)
        
        # 속성으로 체크 (router, experts 등)
        has_router = hasattr(module, 'router')
        has_experts = hasattr(module, 'experts')
        has_gate = hasattr(module, 'gate')
        
        # G3MoE 특정 체크: router가 G3MoERouter인지 확인
        is_g3moe_router = False
        if has_router:
            router = getattr(module, 'router', None)
            if router is not None:
                router_class_name = router.__class__.__name__
                is_g3moe_router = ('G3MoERouter' in router_class_name or 
                                  'GramSpecMoERouter' in router_class_name or
                                  'GramSpecRouter' in router_class_name or
                                  getattr(router, '_is_gramspec_moe_router', False) or
                                  getattr(router, '_is_g3moe_router', False))
        
        is_moe = (is_moe_by_name or 
                  (has_router and has_experts) or  # router와 experts 둘 다 있으면 MoE
                  (is_g3moe_router and has_experts) or  # G3MoE router + experts
                  has_gate)
        
        return is_moe
    
    def _register_vision_hooks(self):
        """Vision tower와 projector에 forward hooks 등록"""
        if self.model is None:
            return
        
        # Vision tower 찾기
        vision_tower = None
        projector = None
        vision_tower_name = None
        projector_name = None
        
        # G3MoE 모델 구조에 맞춰 vision_tower와 multi_modal_projector 찾기
        if hasattr(self.model, 'vision_tower'):
            vision_tower = self.model.vision_tower
            vision_tower_name = 'vision_tower'
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'vision_tower'):
            vision_tower = self.model.model.vision_tower
            vision_tower_name = 'model.vision_tower'
        
        if hasattr(self.model, 'multi_modal_projector'):
            projector = self.model.multi_modal_projector
            projector_name = 'multi_modal_projector'
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'multi_modal_projector'):
            projector = self.model.model.multi_modal_projector
            projector_name = 'model.multi_modal_projector'
        
        # Vision 모듈 정보 저장 (requires_grad 체크용)
        self.vision_modules_info = {
            'vision_tower': {'module': vision_tower, 'name': vision_tower_name},
            'projector': {'module': projector, 'name': projector_name}
        }
        
        # Vision tower hook 등록
        if vision_tower is not None:
            def vision_tower_hook(module, input, output):
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
                # 디버그: hook이 호출되는지 확인 (처음 몇 번만)
                if not hasattr(self, '_hook_call_count'):
                    self._hook_call_count = {}
                if layer_name not in self._hook_call_count:
                    self._hook_call_count[layer_name] = 0
                self._hook_call_count[layer_name] += 1
                # if self._hook_call_count[layer_name] <= 3 and self.log_to_console:
                #     self._log_debug(f"🔍 Hook called for {layer_name} (call #{self._hook_call_count[layer_name]})")
                
                # t-SNE용 데이터 수집 (메모리 절약을 위해 최근 step만)
                # input[0]은 hidden states (MoE layer 입력)
                if isinstance(input, tuple) and len(input) > 0:
                    hidden_states = input[0]
                    if torch.is_tensor(hidden_states) and hidden_states.numel() > 0:
                        # CPU로 이동하고 flatten (메모리 절약)
                        hidden_states_cpu = hidden_states.detach().to('cpu', non_blocking=True)
                        # [batch, seq, hidden_dim] -> [batch*seq, hidden_dim]
                        if hidden_states_cpu.dim() == 3:
                            hidden_states_flat = hidden_states_cpu.reshape(-1, hidden_states_cpu.size(-1))
                            # 샘플링하여 메모리 절약 (최대 1000개 토큰만)
                            if hidden_states_flat.size(0) > 1000:
                                indices = torch.randperm(hidden_states_flat.size(0))[:1000]
                                hidden_states_flat = hidden_states_flat[indices]
                            self.tsne_data_buffer[layer_name]['hidden_states'].append(hidden_states_flat)

                routing_info = self._extract_routing_info(module, input, output)
                if routing_info:
                    # if self._hook_call_count[layer_name] <= 3 and self.log_to_console:
                    #     self._log_debug(f"  ✅ Extracted routing info: {list(routing_info.keys())}")
                    # Store only lightweight, CPU-detached summaries to avoid GPU memory growth
                    lightweight_entry = {}
                    if 'expert_assignments' in routing_info and routing_info['expert_assignments'] is not None:
                        ea = routing_info['expert_assignments']
                        if torch.is_tensor(ea):
                            ea = ea.detach().to('cpu', non_blocking=True)
                            # 1차원으로 확실히 변환 (bincount 요구사항)
                            if ea.dim() > 1:
                                ea = ea.flatten()
                            elif ea.dim() == 0:
                                ea = ea.unsqueeze(0)
                            if ea.dim() != 1:
                                ea = ea.view(-1)
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
                    # G3MoE specific metrics
                    if 'speciality_loss' in routing_info and routing_info['speciality_loss'] is not None:
                        val = routing_info['speciality_loss']
                        if torch.is_tensor(val):
                            val = val.detach().to('cpu')
                        lightweight_entry['speciality_loss'] = val
                    if 'cosine_similarities' in routing_info and routing_info['cosine_similarities'] is not None:
                        val = routing_info['cosine_similarities']
                        if torch.is_tensor(val):
                            val = val.detach().to('cpu')
                        lightweight_entry['cosine_similarities'] = val
                    if 'expression_loss' in routing_info and routing_info['expression_loss'] is not None:
                        val = routing_info['expression_loss']
                        if torch.is_tensor(val):
                            val = val.detach().to('cpu')
                        lightweight_entry['expression_loss'] = val
                    self.layer_outputs[layer_name] = lightweight_entry
                    
                    # t-SNE용 expert assignments 저장
                    if 'expert_assignments' in lightweight_entry:
                        ea = lightweight_entry['expert_assignments']
                        if torch.is_tensor(ea) and ea.numel() > 0:
                            # 샘플링하여 메모리 절약
                            if ea.size(0) > 1000:
                                indices = torch.randperm(ea.size(0))[:1000]
                                ea = ea[indices]
                            self.tsne_data_buffer[layer_name]['expert_assignments'].append(ea)
                    
                    # 디버깅 로그는 항상 출력 (step 정보 제거)
                    # self._log_debug(f"{layer_name}: extracted {list(routing_info.keys())}")
                else:
                    if self._hook_call_count[layer_name] <= 3 and self.log_to_console:
                        # 디버그: 왜 routing_info가 None인지 확인
                        has_last_selected = hasattr(module, 'last_selected_experts')
                        output_is_tuple = isinstance(output, tuple)
                        output_len = len(output) if output_is_tuple else 0
                        self._log_debug(f"  ❌ No routing info extracted for {layer_name}")
                        self._log_debug(f"     - has last_selected_experts: {has_last_selected}")
                        if has_last_selected:
                            se = module.last_selected_experts
                            self._log_debug(f"     - last_selected_experts shape: {se.shape if torch.is_tensor(se) else type(se)}")
                        self._log_debug(f"     - output is tuple: {output_is_tuple}, len: {output_len}")
                        if output_is_tuple and len(output) >= 2:
                            router_info = output[-1]
                            self._log_debug(f"     - router_info type: {type(router_info)}, is tuple: {isinstance(router_info, tuple)}")
                            if isinstance(router_info, tuple):
                                self._log_debug(f"     - router_info len: {len(router_info)}")
            except Exception as e:
                self._log_debug(f"Warning: Failed to extract routing info from {layer_name}: {e}")
        return hook_fn
    
    @torch.no_grad()
    def _extract_routing_info(self, module, input, output):
        """모듈에서 라우팅 정보 추출 (치명적 버그 수정: router의 last_xxx 속성 직접 읽기)"""
        routing_info = {}
        # Lightweight mode: avoid retaining large tensors by default
        lightweight = True

        # ===== 우선순위 1: Router에서 직접 추출 (가장 안정적) =====
        router = getattr(module, 'router', None)
        if router is not None:
            # Router의 last_xxx 속성들이 실제로 존재함 (코드 확인 완료)
            if hasattr(router, 'last_selected_experts') and router.last_selected_experts is not None:
                selected_experts = router.last_selected_experts
                # selected_experts: [batch*seq, top_k] 형태
                if selected_experts.dim() == 2:
                    selected_experts_flat = selected_experts.flatten()
                    routing_info['expert_assignments'] = selected_experts_flat
                else:
                    routing_info['expert_assignments'] = selected_experts.flatten() if selected_experts.dim() > 0 else selected_experts
            
            if hasattr(router, 'last_routing_weights') and router.last_routing_weights is not None:
                routing_weights = router.last_routing_weights
                if routing_weights.dim() == 2:
                    routing_info['routing_probs'] = routing_weights.flatten()
                else:
                    routing_info['routing_probs'] = routing_weights.flatten() if routing_weights.dim() > 0 else routing_weights
            
            if hasattr(router, 'num_experts'):
                routing_info['num_experts'] = router.num_experts
            elif hasattr(router, 'last_num_experts'):
                routing_info['num_experts'] = router.last_num_experts

        # ===== 우선순위 2: 모듈에서 직접 저장된 정보 (fallback) =====
        if 'expert_assignments' not in routing_info and hasattr(module, 'last_selected_experts'):
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
        
        # ===== Router에서 Loss 메트릭 직접 추출 (치명적 버그 수정) =====
        if router is not None:
            # Router의 last_xxx 속성에서 loss 메트릭 추출
            if hasattr(router, 'last_speciality_loss') and router.last_speciality_loss is not None:
                val = router.last_speciality_loss
                if torch.is_tensor(val):
                    val = val.detach().to('cpu')
                routing_info['speciality_loss'] = val
            elif hasattr(router, 'last_ortho_loss') and router.last_ortho_loss is not None:
                # ortho_loss를 speciality_loss로도 사용
                val = router.last_ortho_loss
                if torch.is_tensor(val):
                    val = val.detach().to('cpu')
                routing_info['speciality_loss'] = val
            
            if hasattr(router, 'last_cosine_similarities') and router.last_cosine_similarities is not None:
                val = router.last_cosine_similarities
                if torch.is_tensor(val):
                    val = val.detach().to('cpu')
                routing_info['cosine_similarities'] = val
            
            if hasattr(router, 'last_expression_reg_loss') and router.last_expression_reg_loss is not None:
                val = router.last_expression_reg_loss
                if torch.is_tensor(val):
                    val = val.detach().to('cpu')
                routing_info['expression_loss'] = val
        
        # 실제 G3MoE/GRIN 모델 구조에 맞춘 추출 (우선순위: output에서 직접 추출)
        # G3MoEGRINMoE output: (hidden_states, (routing_probs_full, hn, speciality_loss, cosine_similarities, expression_loss))
        # G3MoEDecoderLayer output: (hidden_states, (self_attn_weights?), (router_logits, hn, speciality_loss, cosine_similarities, expression_loss))
        if isinstance(output, tuple) and len(output) >= 2:
            hidden_states = output[0]
            router_info_tuple = output[-1]  # 마지막 요소가 routing info 튜플
            
            # G3MoE nested tuple 구조 파싱
            if isinstance(router_info_tuple, tuple) and len(router_info_tuple) >= 5:
                routing_probs_full = router_info_tuple[0]  # 실제로는 routing_probs_full (softmax된 확률)
                # hn = router_info_tuple[1]  # 사용 안 함
                speciality_loss = router_info_tuple[2]
                cosine_similarities = router_info_tuple[3]
                expression_loss = router_info_tuple[4]
                
                # ✅ routing_probs_full에서 expert assignments 추출 (last_selected_experts보다 우선)
                # hook이 forward 중에 호출되므로 output에서 직접 추출하는 것이 더 안정적
                if routing_probs_full is not None and torch.is_tensor(routing_probs_full):
                    # routing_probs_full: [batch*seq, num_experts] 형태의 softmax 확률
                    if routing_probs_full.dim() >= 2:
                        # top-k를 고려하여 top-1 expert 선택 (argmax)
                        expert_assignments = routing_probs_full.argmax(dim=-1)  # [batch*seq]
                        # last_selected_experts가 없거나 이미 추출한 경우에만 사용
                        if 'expert_assignments' not in routing_info:
                            routing_info['expert_assignments'] = expert_assignments.flatten()
                        # routing_probs는 항상 업데이트 (entropy 계산용)
                        routing_info['routing_probs'] = routing_probs_full.flatten()
                        # num_experts 정보도 추출
                        if 'num_experts' not in routing_info:
                            routing_info['num_experts'] = routing_probs_full.size(-1)
                    else:
                        expert_assignments = routing_probs_full.argmax(dim=-1)
                        if 'expert_assignments' not in routing_info:
                            routing_info['expert_assignments'] = expert_assignments
                        routing_info['routing_probs'] = routing_probs_full
                        if 'num_experts' not in routing_info and hasattr(module, 'num_experts'):
                            routing_info['num_experts'] = module.num_experts
                
                # Loss 메트릭 저장 (CPU로 이동)
                if speciality_loss is not None:
                    val = speciality_loss.detach().to('cpu') if torch.is_tensor(speciality_loss) else speciality_loss
                    routing_info['speciality_loss'] = val
                if cosine_similarities is not None:
                    val = cosine_similarities.detach().to('cpu') if torch.is_tensor(cosine_similarities) else cosine_similarities
                    routing_info['cosine_similarities'] = val
                if expression_loss is not None:
                    val = expression_loss.detach().to('cpu') if torch.is_tensor(expression_loss) else expression_loss
                    routing_info['expression_loss'] = val
            elif isinstance(router_info_tuple, tuple) and len(router_info_tuple) > 0:
                # 다른 형태의 튜플 (기존 호환성 유지)
                router_logits = router_info_tuple[0]
                if router_logits is not None and torch.is_tensor(router_logits):
                    # Compute expert assignments cheaply; skip storing full probs/logits
                    if 'expert_assignments' not in routing_info:
                        expert_assignments = router_logits.argmax(dim=-1)
                        routing_info['expert_assignments'] = expert_assignments.flatten() if expert_assignments.dim() > 1 else expert_assignments
                    if not lightweight:
                        routing_probs = torch.nn.functional.softmax(router_logits, dim=-1)
                        routing_info['routing_probs'] = routing_probs.flatten() if routing_probs.dim() > 1 else routing_probs
                        routing_info['gate_logits'] = router_logits
            else:
                # 단일 텐서인 경우
                router_logits = router_info_tuple
                if router_logits is not None and torch.is_tensor(router_logits):
                    if 'expert_assignments' not in routing_info:
                        expert_assignments = router_logits.argmax(dim=-1)
                        routing_info['expert_assignments'] = expert_assignments.flatten() if expert_assignments.dim() > 1 else expert_assignments
                    if not lightweight:
                        routing_probs = torch.nn.functional.softmax(router_logits, dim=-1)
                        routing_info['routing_probs'] = routing_probs.flatten() if routing_probs.dim() > 1 else routing_probs
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
        
        # G3MoERouter EMA 수동 업데이트 (Gradient Checkpointing 호환)
        if self.model is not None:
            for module in self.model.modules():
                if hasattr(module, 'update_expert_load_ema') and hasattr(module, 'last_current_load') and module.last_current_load is not None:
                    module.update_expert_load_ema(module.last_current_load)
                    module.last_current_load = None

        # 현재 step 추적 (디버그 메시지 로깅용)
        self._current_step = current_step

        # 강제 모든 랭크 로깅 허용 및 hook 실패 시 모델 상태에서 수집하여 항상 지표 산출
        if not self.layer_outputs:
            if self.log_to_console:
                self._log_debug(f"⚠️ Step {current_step}: No layer_outputs from hooks. Attempting to collect from model state...")
                # Hook 호출 횟수 확인
                if hasattr(self, '_hook_call_count'):
                    total_calls = sum(self._hook_call_count.values())
                    self._log_debug(f"   - Total hook calls so far: {total_calls}")
                    if total_calls == 0:
                        self._log_debug(f"   - ⚠️ WARNING: Hooks were never called! Check if model forward is being executed.")
                    else:
                        self._log_debug(f"   - Hook calls per layer: {dict(list(self._hook_call_count.items())[:5])}")
            collected = self._collect_from_model_state()
            if not collected:
                if self.log_to_console:
                    self._log_debug(f"❌ Step {current_step}: No routing info captured via hooks or model state.")
                    self._log_debug(f"   - Hooks count: {len(self.hooks)}")
                    self._log_debug(f"   - Model is None: {self.model is None}")
                    if self.model is not None:
                        # 모델에서 MoE 레이어 찾기 시도
                        moe_layers_found = []
                        for name, module in self.model.named_modules():
                            if self._is_moe_layer(module):
                                moe_layers_found.append(name)
                                # 첫 번째 MoE 레이어에서 last_selected_experts 확인
                                if len(moe_layers_found) == 1 and hasattr(module, 'last_selected_experts'):
                                    se = module.last_selected_experts
                                    self._log_debug(f"   - Sample MoE layer '{name}' has last_selected_experts: {se.shape if torch.is_tensor(se) else type(se)}")
                        self._log_debug(f"   - MoE layers in model: {len(moe_layers_found)}")
                        if moe_layers_found:
                            self._log_debug(f"   - Sample MoE layers: {moe_layers_found[:3]}")
            else:
                self.layer_outputs.update(collected)
                if self.log_to_console:
                    self._log_debug(f"✅ Step {current_step}: Collected {len(collected)} routing info from model state")
                    self._log_debug(f"   - Collected layer names: {list(collected.keys())[:3]}")
        else:
            if self.log_to_console and current_step % 10 == 0:  # 10 step마다만 로그
                self._log_debug(f"✅ Step {current_step}: layer_outputs has {len(self.layer_outputs)} entries")
                # 실제 MoE 레이어 수와 비교
                if self.model is not None:
                    moe_layers_in_model = []
                    for name, module in self.model.named_modules():
                        if self._is_moe_layer(module):
                            moe_layers_in_model.append(name)
                    if len(moe_layers_in_model) != len(self.layer_outputs):
                        self._log_debug(f"   ⚠️ Mismatch: {len(moe_layers_in_model)} MoE layers in model, but {len(self.layer_outputs)} in layer_outputs")
                        missing_layers = [l for l in moe_layers_in_model if l not in self.layer_outputs]
                        if missing_layers:
                            self._log_debug(f"   ⚠️ Missing layers in layer_outputs: {missing_layers[:5]}")
                        # 누락된 레이어는 모델 상태에서 수집 시도
                        if missing_layers:
                            collected = self._collect_from_model_state()
                            for layer_name in missing_layers:
                                if layer_name in collected:
                                    self.layer_outputs[layer_name] = collected[layer_name]
                                    self._log_debug(f"   ✅ Collected missing layer '{layer_name}' from model state")

        # 메트릭 계산
        step_metrics = self._calculate_step_metrics()
        # wrapper용 last_metrics 저장
        self.last_metrics = step_metrics

        # 디버그: 메트릭이 비어있는지 확인
        if not step_metrics:
            if self.log_to_console:
                self._log_debug(f"Step {current_step}: step_metrics is empty! layer_outputs: {len(self.layer_outputs)}")
                if self.layer_outputs:
                    # layer_outputs는 있는데 메트릭이 없는 경우 - 첫 번째 레이어 확인
                    first_layer = list(self.layer_outputs.keys())[0]
                    first_data = self.layer_outputs[first_layer]
                    self._log_debug(f"   - First layer '{first_layer}' data keys: {list(first_data.keys())}")
                    if 'expert_assignments' in first_data:
                        ea = first_data['expert_assignments']
                        self._log_debug(f"   - expert_assignments: shape={ea.shape if torch.is_tensor(ea) else 'N/A'}, numel={ea.numel() if torch.is_tensor(ea) else 'N/A'}")

        # _log_metrics를 매 step마다 호출하여 log_data 생성
        self._log_metrics(step_metrics, current_step)

        # ✅ on_step_end에서는 pending에만 저장, wandb 로깅은 on_log에서 Trainer와 함께 처리
        # (step 충돌 방지를 위해 직접 wandb 로깅하지 않음)
        if hasattr(self, 'last_log_data') and self.last_log_data:
            # pending에 저장 (on_log에서 Trainer의 WandbCallback과 함께 로깅)
            self.pending_metrics[current_step] = self.last_log_data.copy()
            
            if self.log_to_console and (current_step % 10 == 0 or current_step <= 5):
                moe_metrics = {
                    k: v for k, v in self.last_log_data.items() 
                    if (k.startswith('moe/') or 
                        k.startswith('multi_modality/') or 
                        k.startswith('train/router/'))
                }
                self._log_debug(f"✅ Step {current_step}: stored {len(moe_metrics)} MoE metrics in pending (will be logged in on_log)")
        else:
            if self.log_to_console and (current_step % 10 == 0 or current_step <= 5):
                self._log_debug(f"⚠️ Step {current_step}: no last_log_data to store")
                self._log_debug(f"   - has last_log_data attr: {hasattr(self, 'last_log_data')}")
                if hasattr(self, 'last_log_data'):
                    self._log_debug(f"   - last_log_data is None/empty: {self.last_log_data is None or not self.last_log_data}")
                # step_metrics 확인
                if hasattr(self, 'last_metrics'):
                    self._log_debug(f"   - last_metrics: {self.last_metrics}")

        # 히트맵 생성
        if current_step % self.log_heatmap_every == 0:
            self._generate_heatmaps(current_step)
        
        # t-SNE 시각화 생성
        if current_step % self.log_tsne_every == 0:
            self._generate_tsne_visualizations(current_step)

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
            self._log_debug(f"[MoE Generation] Logging generations at step {current_step}")
            self._log_generations(current_step)

    def _get_tokenizer_ids(self, tokenizer):
        """Processor 또는 Tokenizer에서 pad_token_id와 eos_token_id를 안전하게 가져오기"""
        # Processor 객체인 경우 실제 토크나이저에 접근
        actual_tokenizer = tokenizer
        if hasattr(tokenizer, 'tokenizer'):
            actual_tokenizer = tokenizer.tokenizer
        
        # pad_token_id 가져오기
        pad_token_id = None
        if hasattr(actual_tokenizer, 'pad_token_id') and actual_tokenizer.pad_token_id is not None:
            pad_token_id = actual_tokenizer.pad_token_id
        elif hasattr(tokenizer, 'pad_token') and tokenizer.pad_token is not None:
            # pad_token이 있으면 tokenizer를 통해 id로 변환
            if hasattr(tokenizer, 'convert_tokens_to_ids'):
                pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
            elif hasattr(actual_tokenizer, 'convert_tokens_to_ids'):
                pad_token_id = actual_tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        
        # eos_token_id 가져오기
        eos_token_id = None
        if hasattr(actual_tokenizer, 'eos_token_id') and actual_tokenizer.eos_token_id is not None:
            eos_token_id = actual_tokenizer.eos_token_id
        elif hasattr(tokenizer, 'eos_token') and tokenizer.eos_token is not None:
            # eos_token이 있으면 tokenizer를 통해 id로 변환
            if hasattr(tokenizer, 'convert_tokens_to_ids'):
                eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
            elif hasattr(actual_tokenizer, 'convert_tokens_to_ids'):
                eos_token_id = actual_tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
        
        # fallback: pad_token_id가 없으면 eos_token_id 사용
        if pad_token_id is None:
            pad_token_id = eos_token_id
        
        return pad_token_id, eos_token_id
    
    @torch.no_grad()
    def _test_vlm_capabilities(self, model, tokenizer):
        """VLM 기능 테스트: 멀티모달과 텍스트 전용 케이스 모두 테스트"""
        # tokenizer가 None이면 테스트 스킵
        if tokenizer is None:
            self._log_debug("⚠️ VLM test skipped: tokenizer is None")
            return
        
        # model이 None이면 테스트 스킵
        if model is None:
            self._log_debug("⚠️ VLM test skipped: model is None")
            return
        
        self._log_debug("="*80)
        self._log_debug("🔍 VLM Capabilities Test (Training Start)")
        self._log_debug("="*80)
        
        test_results = {
            "multimodal_tests": [],
            "text_only_tests": [],
            "chat_template_tests": []
        }
        
        original_mode = model.training
        model.eval()
        try:
            # 테스트 1: 멀티모달 (이미지 + 텍스트) 테스트
            self._log_debug("\n📸 Test 1: Multimodal (Image + Text) Generation")
            try:
                sample_image_url = "https://huggingface.co/spaces/merve/chameleon-7b/resolve/main/bee.jpg"
                image = load_image(sample_image_url)
                
                # Chat template 적용 테스트
                multimodal_messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image in Korean."},
                            {"type": "image"}
                        ]
                    }
                ]
                
                # Chat template 적용
                try:
                    chat_template_result = tokenizer.apply_chat_template(
                        multimodal_messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    test_results["chat_template_tests"].append({
                        "type": "multimodal",
                        "status": "success",
                        "template_length": len(chat_template_result)
                    })
                    self._log_debug(f"  ✅ Chat template applied successfully (length: {len(chat_template_result)})")
                except Exception as e:
                    test_results["chat_template_tests"].append({
                        "type": "multimodal",
                        "status": "failed",
                        "error": str(e)
                    })
                    self._log_debug(f"  ❌ Chat template failed: {e}")
                    raise
                
                # 토크나이징 및 생성 테스트
                test_input_text = chat_template_result.replace("<bos>", "")[:-1] if "<bos>" in chat_template_result else chat_template_result
                inputs = tokenizer(
                    text=test_input_text,
                    images=image,
                    return_tensors="pt"
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                pad_token_id, eos_token_id = self._get_tokenizer_ids(tokenizer)
                start_time = time.time()
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    num_return_sequences=1,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    use_cache=True,
                )
                
                input_length = inputs['input_ids'].shape[1]
                generated_text = tokenizer.decode(
                    outputs[0][input_length:],
                    skip_special_tokens=True
                )
                end_time = time.time()
                test_results["multimodal_tests"].append({
                    "status": "success",
                    "generated_length": len(generated_text),
                    "generated_preview": generated_text.strip()[:100]
                })

                self._log_debug(f"  ✅ Multimodal generation successful (time: {end_time - start_time} seconds)")
                self._log_debug(f"     Generated: {generated_text.strip()[:100]}...")
                
            except Exception as e:
                test_results["multimodal_tests"].append({
                    "status": "failed",
                    "error": str(e)
                })
                self._log_debug(f"  ❌ Multimodal test failed: {e}")
                import traceback
                self._log_debug(f"     Traceback: {traceback.format_exc()}")
            
            # 테스트 2: 텍스트 전용 테스트
            self._log_debug("\n📝 Test 2: Text-Only Generation")
            try:
                text_only_messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is the capital of France?"}
                        ]
                    }
                ]
                
                # Chat template 적용 테스트
                try:
                    chat_template_result = tokenizer.apply_chat_template(
                        text_only_messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    test_results["chat_template_tests"].append({
                        "type": "text_only",
                        "status": "success",
                        "template_length": len(chat_template_result)
                    })
                    self._log_debug(f"  ✅ Chat template applied successfully (length: {len(chat_template_result)})")
                except Exception as e:
                    test_results["chat_template_tests"].append({
                        "type": "text_only",
                        "status": "failed",
                        "error": str(e)
                    })
                    self._log_debug(f"  ❌ Chat template failed: {e}")
                    raise
                
                # 토크나이징 및 생성 테스트
                test_input_text = chat_template_result.replace("<bos>", "")[:-1] if "<bos>" in chat_template_result else chat_template_result
                
                # 텍스트 전용이므로 images 파라미터 없이 처리
                if hasattr(tokenizer, 'tokenizer'):
                    # AutoProcessor인 경우
                    inputs = tokenizer(
                        text=test_input_text,
                        return_tensors="pt"
                    )
                else:
                    # AutoTokenizer인 경우
                    inputs = tokenizer(
                        test_input_text,
                        return_tensors="pt"
                    )
                
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                pad_token_id, eos_token_id = self._get_tokenizer_ids(tokenizer)
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    num_return_sequences=1,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    use_cache=True,
                )
                
                input_length = inputs['input_ids'].shape[1]
                generated_text = tokenizer.decode(
                    outputs[0][input_length:],
                    skip_special_tokens=True
                )
                
                test_results["text_only_tests"].append({
                    "status": "success",
                    "generated_length": len(generated_text),
                    "generated_preview": generated_text.strip()[:100]
                })
                self._log_debug(f"  ✅ Text-only generation successful")
                self._log_debug(f"     Generated: {generated_text.strip()[:100]}...")
                
            except Exception as e:
                test_results["text_only_tests"].append({
                    "status": "failed",
                    "error": str(e)
                })
                self._log_debug(f"  ❌ Text-only test failed: {e}")
                import traceback
                self._log_debug(f"     Traceback: {traceback.format_exc()}")
            
            # 테스트 결과 요약
            self._log_debug("\n" + "="*80)
            self._log_debug("📊 VLM Test Summary")
            self._log_debug("="*80)
            
            multimodal_success = any(t.get("status") == "success" for t in test_results["multimodal_tests"])
            text_only_success = any(t.get("status") == "success" for t in test_results["text_only_tests"])
            chat_template_success = all(t.get("status") == "success" for t in test_results["chat_template_tests"])
            
            self._log_debug(f"  Multimodal Test: {'✅ PASS' if multimodal_success else '❌ FAIL'}")
            self._log_debug(f"  Text-Only Test: {'✅ PASS' if text_only_success else '❌ FAIL'}")
            self._log_debug(f"  Chat Template Test: {'✅ PASS' if chat_template_success else '❌ FAIL'}")
            
            # wandb에 로깅
            if self.logger and hasattr(self.logger, 'log'):
                try:
                    wandb_log_data = {
                        'vlm_test/multimodal_success': 1.0 if multimodal_success else 0.0,
                        'vlm_test/text_only_success': 1.0 if text_only_success else 0.0,
                        'vlm_test/chat_template_success': 1.0 if chat_template_success else 0.0,
                    }
                    
                    # 상세 결과도 추가
                    if test_results["multimodal_tests"]:
                        mm_result = test_results["multimodal_tests"][0]
                        if mm_result.get("status") == "success":
                            wandb_log_data['vlm_test/multimodal_generated_length'] = mm_result.get("generated_length", 0)
                    
                    if test_results["text_only_tests"]:
                        to_result = test_results["text_only_tests"][0]
                        if to_result.get("status") == "success":
                            wandb_log_data['vlm_test/text_only_generated_length'] = to_result.get("generated_length", 0)
                    
                    self.logger.log(wandb_log_data, step=0, commit=True)
                    self._log_debug(f"  ✅ Test results logged to wandb")
                except Exception as e:
                    self._log_debug(f"  ⚠️ Failed to log test results to wandb: {e}")
            
            # 전체 테스트 성공 여부
            all_tests_passed = multimodal_success and text_only_success and chat_template_success
            if all_tests_passed:
                self._log_debug("\n✅ All VLM tests passed!")
            else:
                self._log_debug("\n⚠️ Some VLM tests failed. Check the logs above for details.")
            
            self._log_debug("="*80 + "\n")
            
        except Exception as e:
            self._log_debug(f"❌ VLM test error: {e}")
            import traceback
            self._log_debug(traceback.format_exc())
        finally:
            model.train(original_mode)
    
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

            # 처리할 이미지 URL 선택
            images_to_process = sample_image_urls[:self.max_generation_samples]
            
            # 배치 처리로 CPU 사용률 감소 및 속도 향상
            try:
                # 모든 이미지를 먼저 로드
                images = [load_image(url) for url in images_to_process]
                
                # 배치 토크나이징 (이미지가 있는 경우 개별 처리 필요)
                # Vision 모델의 경우 배치 처리가 복잡할 수 있으므로 개별 처리 유지하되 최적화
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                test_input_text = test_input.replace("<bos>", "")[:-1]
                
                for idx, image in enumerate(images):
                    if sample_count >= self.max_generation_samples:
                        break
                    
                    try:
                        # 입력 토큰화
                        inputs = self.tokenizer(
                            text=test_input_text,
                            images=image,
                            return_tensors="pt"
                        )
                        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                        # 생성 실행 (최적화된 설정)
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=50,  # 100 -> 50으로 줄여서 속도 향상
                            num_return_sequences=1,
                            do_sample=False,  # greedy decoding으로 속도 향상 및 CPU 부하 감소
                            pad_token_id=pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            use_cache=True,  # 캐시 사용으로 속도 향상
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

                        # 콘솔 로그 (간소화)
                        if self.log_to_console:
                            self._log_debug(f"Generation sample {sample_count}: {generated_text.strip()[:60]}...")

                    except Exception as e:
                        self._log_debug(f"Error generating for sample {sample_count}: {e}")
                        sample_count += 1
                        continue
                        
            except Exception as batch_error:
                # 배치 처리 실패 시 개별 처리로 fallback
                self._log_debug(f"⚠️ Batch processing failed, falling back to individual: {batch_error}")
                
                for sample_image_url in images_to_process:
                    if sample_count >= self.max_generation_samples:
                        break

                    try:
                        # 입력 토큰화
                        image = load_image(sample_image_url)

                        inputs = self.tokenizer(
                            text=test_input_text,
                            images=image,
                            return_tensors="pt")
                        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

                        # 생성 실행 (최적화된 설정)
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=50,
                            num_return_sequences=1,
                            do_sample=False,
                            pad_token_id=pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            use_cache=True,
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

                        # 콘솔 로그 (간소화)
                        if self.log_to_console:
                            self._log_debug(f"Generation sample {sample_count}: {generated_text.strip()[:60]}...")

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
                if self.logger and hasattr(self.logger, 'log'):
                    try:
                        gen_log_data = {}
                        for i, log_entry in enumerate(generation_logs):
                            gen_log_data[f'generation/step_{current_step}/sample_{i}/prompt'] = log_entry['prompt']
                            gen_log_data[f'generation/step_{current_step}/sample_{i}/generated'] = log_entry['generated'][:200] + "..."
                        self.logger.log(gen_log_data, step=current_step, commit=True)
                    except Exception as e:
                        self._log_debug(f"Warning: Failed to log generation to wandb at step {current_step}: {e}")

            # 모델을 원래 모드로 복원
            self.model.train(original_mode)
            self._log_debug(f"✅ Completed generation logging at step {current_step} ({sample_count} samples)")

        except Exception as e:
            self._log_debug(f"Error during generation logging: {e}")
            # 모델을 다시 training 모드로 전환
            if self.model is not None:
                self.model.train(original_mode)


    @torch.no_grad()
    def _calculate_step_metrics(self):
        """현재 step의 메트릭 계산"""
        metrics = {}
        
        # 디버그: layer_outputs에 포함된 레이어 확인
        if self.log_to_console and len(self.layer_outputs) > 0:
            layer_names = list(self.layer_outputs.keys())
            if len(layer_names) <= 10:
                self._log_debug(f"📊 _calculate_step_metrics: processing {len(layer_names)} layers: {layer_names}")
            else:
                self._log_debug(f"📊 _calculate_step_metrics: processing {len(layer_names)} layers (first 10: {layer_names[:10]})")
        
        for layer_name, routing_info in self.layer_outputs.items():
            layer_metrics = {}
            
            expert_assignments = routing_info.get('expert_assignments')
            routing_probs = routing_info.get('routing_probs')
            
            # ✅ num_experts를 데이터에서 유도 (가능한 한 데이터 기반, 최후에만 초기화 인자 사용)
            num_experts = routing_info.get('num_experts')
            if num_experts is None and routing_probs is not None and torch.is_tensor(routing_probs) and routing_probs.numel() > 0:
                num_experts = int(routing_probs.size(-1))
            if num_experts is None and expert_assignments is not None and torch.is_tensor(expert_assignments) and expert_assignments.numel() > 0:
                try:
                    num_experts = int(expert_assignments.max().item() + 1)
                except Exception:
                    pass
            if num_experts is None:
                num_experts = int(self.num_experts)
            
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
                    
                    # 차원 확인 및 1차원으로 변환 (bincount 요구사항)
                    if expert_assignments.dim() > 1:
                        expert_assignments = expert_assignments.flatten()
                    elif expert_assignments.dim() == 0:
                        # 스칼라인 경우 1차원으로 변환
                        expert_assignments = expert_assignments.unsqueeze(0)
                    
                    # 최종적으로 1차원인지 확인
                    if expert_assignments.dim() != 1:
                        expert_assignments = expert_assignments.view(-1)
                    
                    # 음수 제거 및 long 타입으로 변환
                    expert_assignments = expert_assignments.clamp(min=0).long()
                
                # Expert 사용 분포 계산
                if expert_assignments.numel() == 0:
                    # 빈 텐서인 경우 기본값
                    usage_counts = torch.zeros(num_experts, dtype=torch.long)
                else:
                    # ✅ 올바른 minlength로 bincount 계산
                    inferred_len = int(expert_assignments.max().item() + 1) if expert_assignments.numel() > 0 else num_experts
                    minlength = max(num_experts, inferred_len)
                    usage_counts = torch.bincount(expert_assignments, minlength=minlength)[:num_experts]
                self.expert_usage_history[layer_name].append(usage_counts)
                
                # Layer별 expert usage tracking (실제 검증용)
                if layer_name not in self.layer_expert_usage_counts:
                    self.layer_expert_usage_counts[layer_name] = torch.zeros(num_experts, dtype=torch.long)
                to_add = usage_counts
                if to_add.size(0) != num_experts:
                    to_add = F.pad(to_add, (0, max(0, num_experts - to_add.size(0))))[:num_experts]
                self.layer_expert_usage_counts[layer_name] += to_add
                
                usage_distribution = usage_counts.float() / (usage_counts.sum().clamp_min(1.0))
                
                # 메트릭 계산
                layer_metrics.update({
                    'usage_counts': usage_counts,
                    'usage_distribution': usage_distribution,
                    'expert_cv': torch.std(usage_distribution) / (torch.mean(usage_distribution) + 1e-8),
                    'max_usage_ratio': usage_distribution.max() / (usage_distribution.mean() + 1e-8),
                    'unused_experts': (usage_counts == 0).sum().item(),
                    'active_experts': (usage_counts > 0).sum().item(),
                    # utilization_mean: usage_distribution의 평균 (각 expert의 평균 사용률)
                    'utilization_mean': usage_distribution.mean().item(),
                })
                
                # MaxVio (Global Load Imbalance) calculation
                try:
                    maxvio = self._calculate_maxvio(usage_counts, num_experts)
                    # 유효한 값인 경우에만 추가 (None이 아닌 경우)
                    if maxvio is not None:
                        layer_metrics['maxvio'] = maxvio
                except Exception as e:
                    if self.log_to_console:
                        self._log_debug(f"Warning: Failed to calculate MaxVio for {layer_name}: {e}")
                    # 0으로 fallback하지 않음 - 메트릭을 추가하지 않음
            
            if routing_probs is not None:
                # 라우팅 엔트로피
                if routing_probs.dim() > 2:
                    routing_probs = routing_probs.view(-1, routing_probs.size(-1))
                
                # 각 토큰의 라우팅 엔트로피
                safe_probs = routing_probs.clamp_min(1e-12)
                token_entropy = -torch.sum(safe_probs * torch.log(safe_probs), dim=-1)
                avg_entropy = token_entropy.mean()
                
                layer_metrics.update({
                    'routing_entropy': avg_entropy,
                    'min_entropy': token_entropy.min(),
                    'max_entropy': token_entropy.max(),
                })
                
                # Routing variance calculation
                try:
                    routing_variance = self._calculate_routing_variance(routing_probs)
                    # 유효한 값인 경우에만 추가 (None이 아닌 경우)
                    if routing_variance is not None:
                        layer_metrics['routing_variance'] = routing_variance
                except Exception as e:
                    if self.log_to_console:
                        self._log_debug(f"Warning: Failed to calculate routing variance for {layer_name}: {e}")
                    # 0으로 fallback하지 않음 - 메트릭을 추가하지 않음
                
                # Top-k score gap calculation
                try:
                    topk_gap = self._calculate_topk_score_gap(routing_probs)
                    # 유효한 값인 경우에만 추가 (None이 아닌 경우)
                    if topk_gap is not None:
                        layer_metrics['topk_score_gap'] = topk_gap
                except Exception as e:
                    if self.log_to_console:
                        self._log_debug(f"Warning: Failed to calculate top-k gap for {layer_name}: {e}")
                    # 0으로 fallback하지 않음 - 메트릭을 추가하지 않음
            
            # G3MoE specific metrics
            if 'speciality_loss' in routing_info and routing_info['speciality_loss'] is not None:
                val = routing_info['speciality_loss']
                if torch.is_tensor(val):
                    layer_metrics['speciality_loss'] = val.item() if val.numel() == 1 else val.mean().item()
                else:
                    layer_metrics['speciality_loss'] = float(val)
            
            if 'cosine_similarities' in routing_info and routing_info['cosine_similarities'] is not None:
                val = routing_info['cosine_similarities']
                if torch.is_tensor(val):
                    layer_metrics['cosine_similarities'] = val.item() if val.numel() == 1 else val.mean().item()
                else:
                    layer_metrics['cosine_similarities'] = float(val)
            
            if 'expression_loss' in routing_info and routing_info['expression_loss'] is not None:
                val = routing_info['expression_loss']
                if torch.is_tensor(val):
                    layer_metrics['expression_loss'] = val.item() if val.numel() == 1 else val.mean().item()
                else:
                    layer_metrics['expression_loss'] = float(val)
            
            # Gram matrix orthogonality calculation (if routing_logits available)
            # Check multiple possible keys for routing logits
            routing_logits = (routing_info.get('gate_logits') or 
                            routing_info.get('routing_logits') or
                            routing_info.get('router_logits'))
            if routing_logits is not None and torch.is_tensor(routing_logits) and routing_logits.numel() > 0:
                try:
                    # Ensure routing_logits is on CPU for calculation
                    if routing_logits.is_cuda:
                        routing_logits_cpu = routing_logits.detach().cpu()
                    else:
                        routing_logits_cpu = routing_logits.detach()
                    gram_ortho = self._calculate_gram_orthogonality(routing_logits_cpu)
                    # 유효한 값인 경우에만 추가 (None이 아닌 경우)
                    if gram_ortho is not None:
                        layer_metrics['gram_orthogonality'] = gram_ortho
                except Exception as e:
                    if self.log_to_console and hasattr(self, '_current_step') and self._current_step % 100 == 0:
                        self._log_debug(f"Warning: Failed to calculate Gram orthogonality for {layer_name}: {e}")
                    # Don't add gram_orthogonality if calculation fails - gracefully skip
            
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

    @torch.no_grad()
    def _calculate_maxvio(self, usage_counts, num_experts):
        """Calculate MaxVio (Global Load Imbalance) metric.
        
        MaxVio = max(|load_i - target|) / target
        where target = total_tokens / num_experts
        
        Args:
            usage_counts: Tensor of shape [num_experts] with token counts per expert
            num_experts: Number of experts
            
        Returns:
            MaxVio value as float, or None if calculation is not possible
        """
        if usage_counts is None or usage_counts.numel() == 0:
            return None
        
        total_tokens = usage_counts.sum().float()
        if total_tokens == 0:
            return None
        
        target_per_expert = total_tokens / num_experts
        deviations = torch.abs(usage_counts.float() - target_per_expert)
        maxvio = deviations.max() / (target_per_expert + 1e-8)
        return maxvio.item()
    
    @torch.no_grad()
    def _calculate_routing_variance(self, routing_probs):
        """Calculate routing variance metric.
        
        Measures the variance of routing probability distributions across experts.
        Higher variance indicates more discriminative routing decisions.
        
        Args:
            routing_probs: Tensor of shape [N, num_experts] with routing probabilities
            
        Returns:
            Average variance across tokens as float, or None if calculation is not possible
        """
        if routing_probs is None or routing_probs.numel() == 0:
            return None
        
        # Ensure 2D: [N, num_experts]
        if routing_probs.dim() > 2:
            routing_probs = routing_probs.view(-1, routing_probs.size(-1))
        elif routing_probs.dim() == 1:
            # Single token case, reshape to [1, num_experts]
            routing_probs = routing_probs.unsqueeze(0)
        
        # Need at least 2 experts for variance calculation
        if routing_probs.size(-1) < 2:
            return None
        
        # Calculate variance across experts for each token, then average
        token_variances = torch.var(routing_probs, dim=-1)
        return token_variances.mean().item()
    
    @torch.no_grad()
    def _calculate_topk_score_gap(self, routing_probs):
        """Calculate top-k score gap metric.
        
        Measures the difference between top-1 and top-2 routing scores.
        Larger gap indicates more confident routing decisions.
        
        Args:
            routing_probs: Tensor of shape [N, num_experts] with routing probabilities
            
        Returns:
            Average gap between top-1 and top-2 scores as float, or None if calculation is not possible
        """
        if routing_probs is None or routing_probs.numel() == 0:
            return None
        
        # Ensure 2D: [N, num_experts]
        if routing_probs.dim() > 2:
            routing_probs = routing_probs.view(-1, routing_probs.size(-1))
        elif routing_probs.dim() == 1:
            routing_probs = routing_probs.unsqueeze(0)
        
        # Need at least 2 experts for top-k gap
        if routing_probs.size(-1) < 2:
            return None
        
        # Get top-2 values for each token
        top_k_values, _ = torch.topk(routing_probs, k=min(2, routing_probs.size(-1)), dim=-1)
        
        if top_k_values.size(-1) < 2:
            return None
        
        # Calculate gap: top-1 - top-2
        gap = (top_k_values[:, 0] - top_k_values[:, 1]).mean()
        return gap.item()
    
    @torch.no_grad()
    def _calculate_gram_orthogonality(self, routing_logits):
        """Calculate Gram matrix orthogonality metric.
        
        Measures how orthogonal the expert representations are by computing
        the Frobenius norm of (Gram - I) where Gram is the Gram matrix of
        normalized routing logits.
        
        Args:
            routing_logits: Tensor of shape [N, num_experts, router_dim] or [N, router_dim]
            
        Returns:
            Orthogonality residual (Frobenius norm) as float, or None if calculation is not possible
        """
        if routing_logits is None or routing_logits.numel() == 0:
            return None
        
        # Handle different input shapes
        if routing_logits.dim() == 2:
            # [N, router_dim] - single expert representation per token
            # Normalize
            normalized = F.normalize(routing_logits, p=2, dim=-1)
            # Compute Gram matrix: [N, router_dim] @ [router_dim, N] = [N, N]
            gram = torch.matmul(normalized, normalized.t())
            # Identity matrix
            identity = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
            # Frobenius norm of difference
            diff = gram - identity
            ortho_residual = torch.norm(diff, p='fro').item()
            return ortho_residual
        elif routing_logits.dim() == 3:
            # [N, num_experts, router_dim] - multiple expert representations per token
            # Normalize each expert representation
            normalized = F.normalize(routing_logits, p=2, dim=-1)
            # Compute Gram matrix: [N, num_experts, router_dim] @ [N, router_dim, num_experts] = [N, num_experts, num_experts]
            gram = torch.matmul(normalized, normalized.transpose(-2, -1))
            # Identity matrix for each token
            num_experts = gram.size(-1)
            identity = torch.eye(num_experts, device=gram.device, dtype=gram.dtype)
            identity = identity.unsqueeze(0).expand(gram.size(0), -1, -1)
            # Frobenius norm of difference for each token, then average
            diff = gram - identity
            # Compute Frobenius norm per token: [N, num_experts, num_experts] -> [N]
            token_norms = torch.norm(diff.view(diff.size(0), -1), p='fro', dim=-1)
            ortho_residual = token_norms.mean().item()
            return ortho_residual
        else:
            return None

    @torch.no_grad()
    def _collect_from_model_state(self):
        """
        forward hook이 실행되지 않은 경우를 대비하여, 모델 모듈의 상태 변수에서
        최근 라우팅 정보를 직접 수집한다.
        - 대상: last_selected_experts, last_routing_weights, last_num_experts
        """
        if self.model is None:
            return {}
        collected = {}
        try:
            for name, module in self.model.named_modules():
                has_any = any(
                    hasattr(module, attr) for attr in (
                        'last_selected_experts', 'last_routing_weights', 'last_num_experts'
                    )
                )
                if not has_any:
                    continue
                entry = {}
                if hasattr(module, 'last_selected_experts'):
                    lse = module.last_selected_experts
                    if torch.is_tensor(lse) and lse.numel() > 0:
                        entry['expert_assignments'] = lse.detach().to('cpu', non_blocking=True)
                if hasattr(module, 'last_routing_weights'):
                    lrw = module.last_routing_weights
                    if torch.is_tensor(lrw) and lrw.numel() > 0:
                        entry['routing_probs'] = lrw.detach().to('cpu', non_blocking=True)
                if hasattr(module, 'last_num_experts'):
                    try:
                        entry['num_experts'] = int(module.last_num_experts)
                    except Exception:
                        pass
                if entry:
                    collected[name] = entry
        except Exception as e:
            if self.log_to_console:
                self._log_debug(f"_collect_from_model_state error: {e}")
        return collected
    
    def _log_metrics(self, metrics, current_step: int):
        """메트릭 로깅"""
        # 디버깅: 메트릭 계산 시작 (wandb에만 기록, console 출력 안 함)
        
        log_data = {}
        
        # 레이어별 메트릭 (moe 카테고리로 분리)
        logged_layers = []
        for layer_name, layer_metrics in metrics.items():
            if layer_name.startswith('_'):
                continue  # 내부 메트릭은 건너뛰기
            logged_layers.append(layer_name)
            for metric_name, value in layer_metrics.items():
                if torch.is_tensor(value) and value.numel() == 1:
                    log_data[f'moe/{layer_name}/{metric_name}'] = value.item()
                elif isinstance(value, (int, float)):
                    log_data[f'moe/{layer_name}/{metric_name}'] = value
        
        # 디버그: 로깅된 레이어 확인
        if self.log_to_console and current_step % 10 == 0:
            self._log_debug(f"📊 _log_metrics at step {current_step}: logged {len(logged_layers)} layers")
            if len(logged_layers) <= 10:
                self._log_debug(f"   Logged layers: {logged_layers}")
            else:
                self._log_debug(f"   Logged layers (first 10): {logged_layers[:10]}")
            # layer_outputs와 비교
            if hasattr(self, 'layer_outputs'):
                layer_outputs_keys = list(self.layer_outputs.keys())
                missing_layers = [l for l in layer_outputs_keys if l not in logged_layers]
                if missing_layers:
                    self._log_debug(f"   ⚠️ Missing layers in metrics: {missing_layers[:10]}")
        
        # 전체 평균 메트릭 (moe 카테고리로 분리)
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
            
            # Utilization mean aggregation
            utilization_mean_values = [m['utilization_mean'] 
                                      for m in metrics.values() 
                                      if isinstance(m, dict) and not isinstance(m, str) and 'utilization_mean' in m]
            if utilization_mean_values:
                log_data['moe/avg_utilization_mean'] = np.mean(utilization_mean_values)
            
            # MaxVio aggregation (None이 아닌 값만 포함)
            maxvio_values = [m['maxvio'] 
                            for m in metrics.values() 
                            if isinstance(m, dict) and not isinstance(m, str) and 'maxvio' in m 
                            and m['maxvio'] is not None]
            if maxvio_values:
                log_data['moe/avg_maxvio'] = np.mean(maxvio_values)
            
            # Routing variance aggregation (None이 아닌 값만 포함)
            routing_variance_values = [m['routing_variance'] 
                                      for m in metrics.values() 
                                      if isinstance(m, dict) and not isinstance(m, str) and 'routing_variance' in m
                                      and m['routing_variance'] is not None]
            if routing_variance_values:
                log_data['moe/avg_routing_variance'] = np.mean(routing_variance_values)
            
            # Top-k score gap aggregation (None이 아닌 값만 포함)
            topk_gap_values = [m['topk_score_gap'] 
                              for m in metrics.values() 
                              if isinstance(m, dict) and not isinstance(m, str) and 'topk_score_gap' in m
                              and m['topk_score_gap'] is not None]
            if topk_gap_values:
                log_data['moe/avg_topk_score_gap'] = np.mean(topk_gap_values)
            
            # G3MoE specific metrics (평균 계산)
            speciality_loss_values = [m['speciality_loss'] 
                                     for m in metrics.values() 
                                     if isinstance(m, dict) and not isinstance(m, str) and 'speciality_loss' in m]
            cosine_similarities_values = [m['cosine_similarities'] 
                                         for m in metrics.values() 
                                         if isinstance(m, dict) and not isinstance(m, str) and 'cosine_similarities' in m]
            expression_loss_values = [m['expression_loss'] 
                                     for m in metrics.values() 
                                     if isinstance(m, dict) and not isinstance(m, str) and 'expression_loss' in m]
            
            if speciality_loss_values:
                log_data['moe/avg_speciality_loss'] = np.mean(speciality_loss_values)
            if cosine_similarities_values:
                log_data['moe/avg_cosine_similarities'] = np.mean(cosine_similarities_values)
            if expression_loss_values:
                log_data['moe/avg_expression_loss'] = np.mean(expression_loss_values)
            
            # Gram orthogonality aggregation (None이 아닌 값만 포함)
            gram_ortho_values = [m['gram_orthogonality'] 
                                for m in metrics.values() 
                                if isinstance(m, dict) and not isinstance(m, str) and 'gram_orthogonality' in m
                                and m['gram_orthogonality'] is not None]
            if gram_ortho_values:
                log_data['moe/avg_gram_orthogonality'] = np.mean(gram_ortho_values)
            
            # Layer-wise balance 메트릭 (실제 검증 지표) - moe 카테고리로 분리
            if '_layer_wise_balance' in metrics:
                balance_metrics = metrics['_layer_wise_balance']
                # 실제로 값이 있는 경우만 로깅 (0으로 fallback하지 않음)
                if 'layer_utilization_cv' in balance_metrics:
                    log_data['moe/validation/layer_utilization_cv'] = balance_metrics['layer_utilization_cv']
                if 'layer_utilization_mean' in balance_metrics:
                    log_data['moe/validation/layer_utilization_mean'] = balance_metrics['layer_utilization_mean']
                if 'layer_entropy_mean' in balance_metrics:
                    log_data['moe/validation/layer_entropy_mean'] = balance_metrics['layer_entropy_mean']
                if 'early_late_utilization_ratio' in balance_metrics:
                    log_data['moe/validation/early_late_ratio'] = balance_metrics['early_late_utilization_ratio']
        
        # Paper 벤치마크 메트릭 추가 (GramSpecAnalyzer가 있는 경우) - moe 카테고리로 분리
        if self.gramspec_analyzer is not None:
            try:
                paper_metrics = self.gramspec_analyzer.get_paper_metrics_summary()
                if paper_metrics:
                    # Load balancing metrics
                    if 'load_balancing' in paper_metrics:
                        lb = paper_metrics['load_balancing']
                        if 'coefficient_of_variation' in lb and lb['coefficient_of_variation'] is not None:
                            log_data['moe/paper/load_balancing/cv'] = lb['coefficient_of_variation']
                        if 'load_imbalance_ratio' in lb and lb['load_imbalance_ratio'] is not None:
                            log_data['moe/paper/load_balancing/imbalance_ratio'] = lb['load_imbalance_ratio']
                        if 'expert_utilization_rate' in lb and lb['expert_utilization_rate'] is not None:
                            log_data['moe/paper/load_balancing/utilization_rate'] = lb['expert_utilization_rate']
                    
                    # Expert specialization metrics
                    if 'expert_specialization' in paper_metrics:
                        es = paper_metrics['expert_specialization']
                        if 'expert_diversity_score' in es and es['expert_diversity_score'] is not None:
                            log_data['moe/paper/expert_specialization/diversity_score'] = es['expert_diversity_score']
                        if 'expert_similarity_mean' in es and es['expert_similarity_mean'] is not None:
                            log_data['moe/paper/expert_specialization/similarity_mean'] = es['expert_similarity_mean']
                        if 'expert_specialization_strength' in es and es['expert_specialization_strength'] is not None:
                            log_data['moe/paper/expert_specialization/specialization_strength'] = es['expert_specialization_strength']
                    
                    # Gram matrix quality
                    if 'gram_matrix_quality' in paper_metrics:
                        gm = paper_metrics['gram_matrix_quality']
                        if 'orthogonality' in gm and gm['orthogonality'] is not None:
                            log_data['moe/paper/gram_matrix/orthogonality'] = gm['orthogonality']
                        if 'orthogonality_std' in gm and gm['orthogonality_std'] is not None:
                            log_data['moe/paper/gram_matrix/orthogonality_std'] = gm['orthogonality_std']
                    
                    # Routing quality
                    if 'routing_quality' in paper_metrics:
                        rq = paper_metrics['routing_quality']
                        if 'routing_confidence' in rq and rq['routing_confidence'] is not None:
                            log_data['moe/paper/routing/confidence'] = rq['routing_confidence']
                        if 'cosine_similarity_mean' in rq and rq['cosine_similarity_mean'] is not None:
                            log_data['moe/paper/routing/cosine_similarity_mean'] = rq['cosine_similarity_mean']
            except Exception as e:
                self._log_debug(f"Warning: Failed to get paper metrics: {e}")
        
        # Vision 모듈 사용 통계 추가 (step별 증가량 계산) - vision 카테고리로 분리
        if hasattr(self, 'prev_vision_stats'):
            step_vision_tower_calls = self.vision_usage_stats['vision_tower_calls'] - self.prev_vision_stats.get('vision_tower_calls', 0)
            step_projector_calls = self.vision_usage_stats['projector_calls'] - self.prev_vision_stats.get('projector_calls', 0)
            step_pixel_values = self.vision_usage_stats['pixel_values_received'] - self.prev_vision_stats.get('pixel_values_received', 0)
            step_image_features = self.vision_usage_stats['image_features_generated'] - self.prev_vision_stats.get('image_features_generated', 0)
            
            # Vision 사용 통계는 항상 로깅 (0이어도) - vision 카테고리로 분리
            log_data['multi_modality/vision_tower_calls_per_step'] = step_vision_tower_calls
            log_data['multi_modality/projector_calls_per_step'] = step_projector_calls
            log_data['multi_modality/pixel_values_per_step'] = step_pixel_values
            log_data['multi_modality/image_features_per_step'] = step_image_features
            
            # 누적 통계도 함께 로깅 - vision 카테고리로 분리
            log_data['multi_modality/vision_tower_calls_total'] = self.vision_usage_stats['vision_tower_calls']
            log_data['multi_modality/projector_calls_total'] = self.vision_usage_stats['projector_calls']
            log_data['multi_modality/pixel_values_total'] = self.vision_usage_stats['pixel_values_received']
            log_data['multi_modality/image_features_total'] = self.vision_usage_stats['image_features_generated']
            
            # Vision 사용률 (이미지가 있는 배치 비율) - vision 카테고리로 분리
            log_data['multi_modality/vision_usage_rate'] = 1.0 if step_vision_tower_calls > 0 else 0.0
            
            # Vision tower 출력 통계 - vision 카테고리로 분리
            if self.vision_tower_outputs:
                recent_outputs = self.vision_tower_outputs[-10:]  # 최근 10개
                log_data['multi_modality/tower_output_mean'] = np.mean([o['mean'] for o in recent_outputs])
                log_data['multi_modality/tower_output_std'] = np.mean([o['std'] for o in recent_outputs])
                log_data['multi_modality/tower_output_min'] = np.min([o['min'] for o in recent_outputs])
                log_data['multi_modality/tower_output_max'] = np.max([o['max'] for o in recent_outputs])
            
            # Projector 출력 통계 - vision 카테고리로 분리
            if self.projector_outputs:
                recent_outputs = self.projector_outputs[-10:]  # 최근 10개
                log_data['multi_modality/projector_output_mean'] = np.mean([o['mean'] for o in recent_outputs])
                log_data['multi_modality/projector_output_std'] = np.mean([o['std'] for o in recent_outputs])
                log_data['multi_modality/projector_output_min'] = np.min([o['min'] for o in recent_outputs])
                log_data['multi_modality/projector_output_max'] = np.max([o['max'] for o in recent_outputs])
            
            # Router의 requires_grad 상태 체크 (MoE upcycling의 핵심) - router 카테고리로 분리
            if self.model is not None:
                router_count = 0
                router_trainable_count = 0
                router_total_params = 0
                router_trainable_params = 0
                
                try:
                    # G3MoERouter 찾기
                    from models.g3moe_model import G3MoERouter
                    for name, module in self.model.named_modules():
                        if (getattr(module, "_is_g3moe_router", False) or 
                            isinstance(module, G3MoERouter)):
                            router_count += 1
                            router_params = list(module.parameters(recurse=True))
                            if router_params:
                                router_total_params += len(router_params)
                                router_trainable = sum(1 for p in router_params if p.requires_grad)
                                router_trainable_params += router_trainable
                                if router_trainable > 0:
                                    router_trainable_count += 1
                    
                    # GramSpecRouter 찾기
                    try:
                        from models.gramspec_moe import GramSpecRouter
                        for name, module in self.model.named_modules():
                            if isinstance(module, GramSpecRouter):
                                router_count += 1
                                router_params = list(module.parameters(recurse=True))
                                if router_params:
                                    router_total_params += len(router_params)
                                    router_trainable = sum(1 for p in router_params if p.requires_grad)
                                    router_trainable_params += router_trainable
                                    if router_trainable > 0:
                                        router_trainable_count += 1
                    except ImportError:
                        pass
                    
                    # Router 통계 로깅 - router 카테고리로 분리
                    if router_count > 0:
                        log_data['train/router/total_routers'] = router_count
                        log_data['train/router/trainable_routers'] = router_trainable_count
                        log_data['train/router/requires_grad'] = 1.0 if router_trainable_count > 0 else 0.0
                        log_data['train/router/trainable_params'] = router_trainable_params
                        log_data['train/router/total_params'] = router_total_params
                        log_data['train/router/trainable_ratio'] = router_trainable_params / max(router_total_params, 1)
                        log_data['train/router/trainable_router_ratio'] = router_trainable_count / max(router_count, 1)
                except Exception as e:
                    process_info = _get_process_info()
                    import traceback
                    error_msg = (
                        f"[MoE Callback] ❌ ERROR in _log_metrics (router requires_grad check):\n"
                        f"  Process: rank={process_info['rank']}, RANK={process_info['RANK']}\n"
                        f"  Step: {current_step}\n"
                        f"  Method: _log_metrics\n"
                        f"  Error: {type(e).__name__}: {str(e)}\n"
                        f"  Traceback:\n{traceback.format_exc()}"
                    )
                    self._log_debug(error_msg)
        
        # log_data가 비어있어도 최소한의 디버그 정보는 로깅 - moe 카테고리로 분리
        if not log_data:
            log_data['moe/no_metrics'] = 1.0
            log_data['moe/layer_outputs_count'] = len(self.layer_outputs)
            log_data['moe/hooks_count'] = len(self.hooks)
            log_data['moe/vision_hooks_count'] = len(self.vision_hooks)
            log_data['moe/metrics_empty'] = 1.0 if not metrics else 0.0
        
        # log_data를 저장하여 Trainer의 logs에 추가할 수 있도록 함
        self.last_log_data = log_data
        
        # 디버그: log_data 생성 확인 (초기 step에서만)
        if self.log_to_console and current_step <= 5:
            self._log_debug(f"✅ _log_metrics at step {current_step}: created {len(log_data)} metrics")
            if log_data:
                sample_keys = list(log_data.keys())[:5]
                self._log_debug(f"   Sample keys: {sample_keys}")
            else:
                self._log_debug(f"   ⚠️ log_data is empty! metrics dict: {list(metrics.keys()) if metrics else 'empty'}")

        # 콘솔 출력 (log_to_console=True일 때만)
        if self.log_to_console:
            self._log_debug(f"Step {current_step} MoE Metrics ({len(log_data)} metrics):")
            for key, value in log_data.items():
                # train/ prefix 제거하여 콘솔에 출력
                display_key = key.replace('train/', '') if key.startswith('train/') else key
                if 'avg_' in display_key or 'total_' in display_key or 'paper/' in display_key or 'router/' in display_key or 'vision/' in display_key or 'moe/' in display_key:
                    if value is not None and isinstance(value, (int, float)):
                        self._log_debug(f"  {display_key}: {value:.4f}")
                    else:
                        self._log_debug(f"  {display_key}: {value}")
    
    def _generate_heatmaps(self, current_step: int):
        """Expert 사용률 히트맵 데이터 생성"""
        # 이미 on_step_end에서 rank 체크하므로 여기서는 생략
        
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
                    if self.log_to_console:
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
                    
                    # Heatmap 데이터를 pending에 저장 (on_log에서 로깅)
                    try:
                        import wandb
                        if current_step not in self.pending_heatmaps:
                            self.pending_heatmaps[current_step] = {}
                        self.pending_heatmaps[current_step][layer_name] = wandb.Image(plt)
                        if self.log_to_console:
                            self._log_debug(f"✅ Generated heatmap for {layer_name} at step {current_step}")
                    except ImportError:
                        if self.log_to_console:
                            self._log_debug(f"⚠️ wandb not available for heatmap generation")
                    except Exception as e:
                        process_info = _get_process_info()
                        import traceback
                        error_msg = (
                            f"[MoE Callback] ❌ ERROR in _generate_heatmaps:\n"
                            f"  Process: rank={process_info['rank']}, RANK={process_info['RANK']}\n"
                            f"  Step: {current_step}\n"
                            f"  Layer: {layer_name}\n"
                            f"  Method: _generate_heatmaps\n"
                            f"  Error: {type(e).__name__}: {str(e)}\n"
                            f"  Traceback:\n{traceback.format_exc()}"
                        )
                        # 에러는 항상 출력
                        print(error_msg)

                    
                    # 파일로 저장
                    if self.save_detailed_logs:
                        try:
                            safe_layer_name = layer_name.replace(".", "_").replace("/", "_")
                            heatmap_path = os.path.join(self.log_dir, f'{safe_layer_name}_heatmap_step_{current_step}.png')
                            plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
                            if self.log_to_console:
                                self._log_debug(f"Heatmap saved to {heatmap_path}")
                        except Exception as e:
                            if self.log_to_console:
                                self._log_debug(f"Warning: Failed to save heatmap: {e}")
                    
                    plt.close()
                    heatmap_created = True
                    
                except Exception as e:
                    import traceback
                    if self.log_to_console:
                        self._log_debug(f"Error creating heatmap for {layer_name}: {e}\n{traceback.format_exc()}")
                    continue
            
            if not heatmap_created and self.log_to_console:
                self._log_debug(f"No heatmaps were created at step {current_step}")
                
        except ImportError as e:
            if self.log_to_console:
                self._log_debug(f"Warning: matplotlib/seaborn not available for heatmap logging: {e}")
        except Exception as e:
            import traceback
            if self.log_to_console:
                self._log_debug(f"Error during heatmap logging: {e}\n{traceback.format_exc()}")
    
    def _generate_tsne_visualizations(self, current_step: int):
        """Layer별 t-SNE 시각화 생성 (expert clustering 시각화)"""
        if not self.tsne_data_buffer:
            if self.log_to_console:
                self._log_debug(f"No t-SNE data available at step {current_step}")
            return
        
        try:
            from sklearn.manifold import TSNE
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            tsne_created = False
            
            for layer_name, buffer in self.tsne_data_buffer.items():
                hidden_states_list = buffer['hidden_states']
                expert_assignments_list = buffer['expert_assignments']
                
                if not hidden_states_list or not expert_assignments_list:
                    continue
                
                # 최근 데이터만 사용 (메모리 절약)
                recent_hidden = list(hidden_states_list)[-10:]  # 최근 10개 step
                recent_experts = list(expert_assignments_list)[-10:]
                
                if not recent_hidden or not recent_experts:
                    continue
                
                try:
                    # 데이터 결합
                    all_hidden = torch.cat(recent_hidden, dim=0).numpy()  # [num_tokens, hidden_dim]
                    all_experts = torch.cat(recent_experts, dim=0).numpy()  # [num_tokens]
                    
                    # 샘플링 (t-SNE 계산 비용 절감)
                    if len(all_hidden) > self.tsne_sample_size:
                        indices = np.random.choice(len(all_hidden), self.tsne_sample_size, replace=False)
                        sampled_hidden = all_hidden[indices]
                        sampled_experts = all_experts[indices]
                    else:
                        sampled_hidden = all_hidden
                        sampled_experts = all_experts
                    
                    if len(sampled_hidden) < 10:  # 최소 샘플 수 확인
                        if self.log_to_console:
                            self._log_debug(f"Insufficient samples for {layer_name} t-SNE: {len(sampled_hidden)}")
                        continue
                    
                    # t-SNE 계산
                    if self.log_to_console:
                        self._log_debug(f"Computing t-SNE for {layer_name} with {len(sampled_hidden)} samples...")
                    
                    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(sampled_hidden) - 1))
                    embeddings_2d = tsne.fit_transform(sampled_hidden)
                    
                    # 시각화
                    num_experts = int(sampled_experts.max() + 1) if len(sampled_experts) > 0 else self.num_experts
                    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
                    
                    # Expert별로 색상 구분
                    colors = plt.cm.tab20(np.linspace(0, 1, num_experts))
                    for expert_id in range(num_experts):
                        mask = sampled_experts == expert_id
                        if mask.sum() > 0:
                            ax.scatter(
                                embeddings_2d[mask, 0],
                                embeddings_2d[mask, 1],
                                label=f'Expert {expert_id}',
                                alpha=0.6,
                                s=20,
                                c=[colors[expert_id % len(colors)]]
                            )
                    
                    ax.set_title(f'{layer_name} Token Clustering by Expert (t-SNE)\nStep {current_step}', fontsize=14)
                    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
                    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                    ax.grid(alpha=0.3)
                    plt.tight_layout()
                    
                    # wandb에 로깅
                    try:
                        import wandb
                        if wandb.run is not None:
                            if current_step not in self.pending_heatmaps:
                                self.pending_heatmaps[current_step] = {}
                            self.pending_heatmaps[current_step][f'{layer_name}_tsne'] = wandb.Image(plt)
                            if self.log_to_console:
                                self._log_debug(f"✅ Generated t-SNE visualization for {layer_name} at step {current_step}")
                    except ImportError:
                        if self.log_to_console:
                            self._log_debug(f"⚠️ wandb not available for t-SNE visualization")
                    except Exception as e:
                        if self.log_to_console:
                            self._log_debug(f"Warning: Failed to log t-SNE to wandb: {e}")
                    
                    # 파일로 저장
                    if self.save_detailed_logs:
                        try:
                            safe_layer_name = layer_name.replace(".", "_").replace("/", "_")
                            tsne_path = os.path.join(self.log_dir, f'{safe_layer_name}_tsne_step_{current_step}.png')
                            plt.savefig(tsne_path, dpi=150, bbox_inches='tight')
                            if self.log_to_console:
                                self._log_debug(f"t-SNE visualization saved to {tsne_path}")
                        except Exception as e:
                            if self.log_to_console:
                                self._log_debug(f"Warning: Failed to save t-SNE visualization: {e}")
                    
                    plt.close()
                    tsne_created = True
                    
                except Exception as e:
                    import traceback
                    if self.log_to_console:
                        self._log_debug(f"Error creating t-SNE for {layer_name}: {e}\n{traceback.format_exc()}")
                    continue
            
            if not tsne_created and self.log_to_console:
                self._log_debug(f"No t-SNE visualizations were created at step {current_step}")
                
        except ImportError as e:
            if self.log_to_console:
                self._log_debug(f"Warning: sklearn/matplotlib not available for t-SNE visualization: {e}")
        except Exception as e:
            import traceback
            if self.log_to_console:
                self._log_debug(f"Error during t-SNE visualization: {e}\n{traceback.format_exc()}")
    
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
        # 이미 on_step_end에서 rank 체크하므로 여기서는 생략
            
        for alert in alerts:
            self.alerts_history.append({
                'step': current_step,
                'timestamp': time.time(),
                **alert
            })
            
            # 경고 메시지는 log_to_console일 때만 출력
            if self.log_to_console:
                self._log_debug(f"⚠️  MoE Alert at step {current_step}: {alert['message']}")
            
            # Alert 데이터를 pending에 저장 (on_log에서 로깅)
            if current_step not in self.pending_alerts:
                self.pending_alerts[current_step] = []
            self.pending_alerts[current_step].append({
                'type': alert["type"],
                'layer': alert["layer"],
                'severity': alert.get('severity', 1)
            })
    
    def _save_detailed_log(self, metrics, current_step: int):
        """상세 로그 저장"""
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
        """훈련 시작 시 모델과 토크나이저 등록 및 VLM 테스트"""
        # ✅ tokenizer가 None이면 여러 방법으로 가져오기 시도
        if tokenizer is None:
            # 방법 1: kwargs에서 trainer 가져오기
            trainer = kwargs.get('trainer')
            if trainer is not None:
                # SFTTrainer는 processing_class를 사용
                if hasattr(trainer, 'processing_class') and trainer.processing_class is not None:
                    tokenizer = trainer.processing_class
                    self.torch_callback._log_debug("✅ Retrieved tokenizer from trainer.processing_class")
                # 일반 Trainer는 tokenizer 속성 사용
                elif hasattr(trainer, 'tokenizer') and trainer.tokenizer is not None:
                    tokenizer = trainer.tokenizer
                    self.torch_callback._log_debug("✅ Retrieved tokenizer from trainer.tokenizer")
            
            # 방법 2: kwargs에서 직접 tokenizer 찾기
            if tokenizer is None:
                tokenizer = kwargs.get('tokenizer') or kwargs.get('processing_class')
                if tokenizer is not None:
                    self.torch_callback._log_debug("✅ Retrieved tokenizer from kwargs")
            
            # 방법 3: torch_callback에 이미 저장된 tokenizer 사용
            if tokenizer is None and hasattr(self.torch_callback, 'tokenizer') and self.torch_callback.tokenizer is not None:
                tokenizer = self.torch_callback.tokenizer
                self.torch_callback._log_debug("✅ Using tokenizer from torch_callback.tokenizer")
            
            if tokenizer is None:
                self.torch_callback._log_debug("⚠️ Could not retrieve tokenizer from any source")
                self.torch_callback._log_debug(f"   - kwargs keys: {list(kwargs.keys())}")
                if trainer is not None:
                    self.torch_callback._log_debug(f"   - trainer has processing_class: {hasattr(trainer, 'processing_class')}")
                    if hasattr(trainer, 'processing_class'):
                        self.torch_callback._log_debug(f"   - trainer.processing_class: {trainer.processing_class}")
                    self.torch_callback._log_debug(f"   - trainer has tokenizer: {hasattr(trainer, 'tokenizer')}")
                    if hasattr(trainer, 'tokenizer'):
                        self.torch_callback._log_debug(f"   - trainer.tokenizer: {trainer.tokenizer}")
        
        if model is not None and not self._model_registered:
            # DeepSpeed 래핑된 모델 처리
            actual_model = model
            if hasattr(model, 'module'):  # DeepSpeed 래핑
                actual_model = model.module
                self.torch_callback._log_debug("⚠️ Detected DeepSpeed wrapped model, using model.module")
            
            self.torch_callback.register_model(actual_model, tokenizer)
            self._model_registered = True
            self.torch_callback._log_debug(f"✅ MoE monitoring registered for model with {len(self.torch_callback.hooks)} MoE layers")
            
            # 디버그: 등록된 MoE 레이어 이름 출력
            if self.torch_callback.hooks:
                moe_layer_names = []
                for name, module in actual_model.named_modules():
                    if self.torch_callback._is_moe_layer(module):
                        moe_layer_names.append(name)
                self.torch_callback._log_debug(f"📋 Registered MoE layers: {moe_layer_names[:5]}..." if len(moe_layer_names) > 5 else f"📋 Registered MoE layers: {moe_layer_names}")
            else:
                self.torch_callback._log_debug("❌ WARNING: No MoE layers detected! Check model structure.")

            if self.torch_callback.enable_generation_logging:
                if tokenizer is not None:
                    self.torch_callback._log_debug("Generation logging enabled with tokenizer")
                else:
                    self.torch_callback._log_debug("Warning: Generation logging enabled but no tokenizer provided")
        
        # wandb.run이 초기화되어 있으면 logger를 wandb.run으로 설정
        # 주의: on_train_begin 시점에는 wandb.run이 아직 None일 수 있음 (Trainer가 나중에 초기화)
        # 실제 로깅은 on_step_end에서 Trainer의 logs를 통해 이루어지므로 여기서는 경고만 출력하지 않음
        try:
            import wandb
            # logger가 wandb 모듈 자체이거나 None인 경우 wandb.run으로 설정
            if (self.torch_callback.logger is None or 
                self.torch_callback.logger == wandb or
                (hasattr(self.torch_callback.logger, '__name__') and self.torch_callback.logger.__name__ == 'wandb')):
                if wandb.run is not None:
                    self.torch_callback.logger = wandb.run
                    self.torch_callback._log_debug("✅ Set MoE callback logger to wandb.run")
                # wandb.run이 None이어도 문제없음 (Trainer의 logs를 통해 로깅됨)
        except ImportError:
            pass
        except Exception as e:
            self.torch_callback._log_debug(f"⚠️ Error setting wandb logger: {e}")
        
        # VLM 테스트: 멀티모달과 텍스트 전용 케이스 모두 테스트
        # tokenizer와 model이 모두 있을 때만 테스트
        actual_model = model
        if hasattr(model, 'module'):  # DeepSpeed 래핑
            actual_model = model.module
        
        # if tokenizer is not None and actual_model is not None:
        #     self.torch_callback._test_vlm_capabilities(actual_model, tokenizer)
        else:
            if tokenizer is None:
                self.torch_callback._log_debug("⚠️ VLM test skipped: tokenizer is None")
            if actual_model is None:
                self.torch_callback._log_debug("⚠️ VLM test skipped: model is None")
    
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
        # PyTorch callback 호출 - Transformers의 global_step 사용
        self.torch_callback.on_step_end(current_step=state.global_step)

        try:
            import deepspeed.accelerator as ds_acc
            ds_acc.get_accelerator().empty_cache()
        except Exception:
            pass
    
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        """Trainer가 로깅할 때 호출 - MoE 메트릭을 logs에 추가하여 Trainer의 WandbCallback이 로깅"""
        # logs가 None이면 빈 dict로 초기화
        if logs is None:
            logs = {}

        # global_step을 logs에 명시적으로 추가 (wandb에서 step 추적용)
        logs['train/global_step'] = float(state.global_step)
        
        # 해당 step의 pending 메트릭 확인
        # on_log는 logging_steps마다 호출되므로, 최근 step의 메트릭을 찾아야 함
        current_metrics = None
        target_step = state.global_step
        
        # 정확한 step의 메트릭이 없으면, 최근 step의 메트릭을 찾음
        if target_step not in self.torch_callback.pending_metrics:
            # 최근 step의 메트릭 찾기 (현재 step 이하의 가장 가까운 step)
            available_steps = [s for s in self.torch_callback.pending_metrics.keys() if s <= target_step]
            if available_steps:
                target_step = max(available_steps)
                current_metrics = self.torch_callback.pending_metrics.get(target_step)
                if self.torch_callback.log_to_console and state.global_step % 10 == 0:
                    self.torch_callback._log_debug(f"📊 Using metrics from step {target_step} for on_log at step {state.global_step}")
            else:
                # 아직 pending 메트릭이 없으면, on_step_end에서 생성된 메트릭을 직접 사용
                if hasattr(self.torch_callback, 'last_log_data') and self.torch_callback.last_log_data:
                    current_metrics = self.torch_callback.last_log_data.copy()
                    if self.torch_callback.log_to_console and state.global_step % 10 == 0:
                        self.torch_callback._log_debug(f"📊 Using last_log_data for step {state.global_step}")
        else:
            current_metrics = self.torch_callback.pending_metrics.get(target_step)
        
        if current_metrics:
            # ✅ logs에 추가 (Trainer의 다른 로깅과 함께)
            logs.update(current_metrics)
            
            # ✅ MoE 메트릭을 wandb에 직접 로깅 (Trainer의 WandbCallback이 일부만 로깅할 수 있으므로)
            # step을 명시하지 않으면 wandb가 자동으로 Trainer의 step을 사용 (충돌 없음)
            try:
                import wandb
                if wandb.run is not None and _is_main_process():
                    # MoE 관련 메트릭만 필터링
                    moe_metrics = {
                        k: v for k, v in current_metrics.items() 
                        if (k.startswith('moe/') or 
                            k.startswith('multi_modality/') or 
                            k.startswith('train/router/'))
                    }
                    
                    if moe_metrics:
                        # ✅ step을 명시하지 않고 로깅 (wandb가 Trainer의 step을 자동으로 사용)
                        # commit=False로 설정하여 Trainer의 로깅과 함께 처리
                        wandb.run.log(moe_metrics, commit=False)
                        
                        if self.torch_callback.log_to_console and state.global_step % 10 == 0:
                            self.torch_callback._log_debug(f"📤 on_log step {state.global_step}: directly logged {len(moe_metrics)} MoE metrics to wandb")
                            if state.global_step <= 5:
                                sample_keys = list(moe_metrics.keys())[:10]
                                self.torch_callback._log_debug(f"   Sample keys: {sample_keys}")
                    
                    # Heatmap/t-SNE는 별도 로깅 (이미지이므로)
                    if state.global_step in self.torch_callback.pending_heatmaps:
                        heatmap_data = self.torch_callback.pending_heatmaps[state.global_step]
                        for layer_name, image in heatmap_data.items():
                            if layer_name.endswith('_tsne'):
                                wandb.run.log({
                                    f'moe/{layer_name}/tsne_visualization': image
                                }, commit=False)
                            else:
                                wandb.run.log({
                                    f'moe/{layer_name}/usage_heatmap': image
                                }, commit=False)
                        del self.torch_callback.pending_heatmaps[state.global_step]
                    
                    # Pending alert 로깅
                    if state.global_step in self.torch_callback.pending_alerts:
                        alert_data = self.torch_callback.pending_alerts[state.global_step]
                        for alert in alert_data:
                            wandb.run.log({
                                f'train/alerts/{alert["type"]}': 1,
                                f'train/alerts/{alert["layer"]}_severity': alert['severity']
                            }, commit=False)
                        del self.torch_callback.pending_alerts[state.global_step]
            except Exception as e:
                if self.torch_callback.log_to_console:
                    import traceback
                    self.torch_callback._log_debug(f"⚠️ Error logging MoE metrics to wandb in on_log: {e}")
                    if state.global_step % 50 == 0:
                        self.torch_callback._log_debug(f"   Traceback: {traceback.format_exc()}")
            
            # 디버그: current_metrics 내용 확인 (10 step마다만)
            if self.torch_callback.log_to_console and state.global_step % 10 == 0:
                moe_keys = [k for k in current_metrics.keys() if k.startswith('moe/') or k.startswith('multi_modality/') or k.startswith('train/router/')]
                self.torch_callback._log_debug(f"📊 on_log step {state.global_step}: total {len(moe_keys)} MoE metrics available")
            
            # 로깅 후 pending 메트릭 제거 (메모리 절약)
            try:
                steps_to_remove = [s for s in self.torch_callback.pending_metrics.keys() if s <= state.global_step]
                for step in steps_to_remove:
                    if step in self.torch_callback.pending_metrics:
                        del self.torch_callback.pending_metrics[step]
            except Exception as e:
                if self.torch_callback.log_to_console:
                    self.torch_callback._log_debug(f"⚠️ Error cleaning up pending metrics: {e}")
        else:
            # 해당 step의 pending 메트릭이 없으면 경고 (log_to_console일 때만)
            if self.torch_callback.log_to_console:
                self.torch_callback._log_debug(f"⚠️ No pending metrics available at step {state.global_step}")
                self.torch_callback._log_debug(f"   - pending_metrics keys: {list(self.torch_callback.pending_metrics.keys())[:10]}")
                self.torch_callback._log_debug(f"   - last_log_data exists: {hasattr(self.torch_callback, 'last_log_data') and self.torch_callback.last_log_data is not None}")
                if hasattr(self.torch_callback, 'last_log_data') and self.torch_callback.last_log_data:
                    self.torch_callback._log_debug(f"   - last_log_data keys: {list(self.torch_callback.last_log_data.keys())[:10]}")
                self.torch_callback._log_debug(f"   - layer_outputs count: {len(self.torch_callback.layer_outputs)}")
            # 최소한의 디버그 정보라도 추가
            logs['moe/callback_error'] = 1.0
            logs['moe/layer_outputs_count'] = len(self.torch_callback.layer_outputs) if hasattr(self.torch_callback, 'layer_outputs') else 0
            logs['moe/hooks_count'] = len(self.torch_callback.hooks) if hasattr(self.torch_callback, 'hooks') else 0
            logs['moe/pending_metrics_count'] = len(self.torch_callback.pending_metrics) if hasattr(self.torch_callback, 'pending_metrics') else 0
    
    def _run_benchmarks(
        self,
        model,
        tokenizer,
        eval_dataloader,
        state: TrainerState,
        args: TrainingArguments,
        **kwargs
    ):
        """벤치마크 실행"""
        if not _is_main_process():
            return
        
        benchmark_results = {}
        output_dir = getattr(self.torch_callback, 'log_dir', './moe_logs')
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. GramSpec MoE Analysis (이미 실행됨, 결과만 정리)
        if GRAMSPEC_ANALYSIS_AVAILABLE and self.torch_callback.gramspec_analyzer is not None:
            try:
                self.torch_callback._log_debug("Running GramSpec MoE Analysis benchmark...")
                analyzer = self.torch_callback.gramspec_analyzer
                aggregated = analyzer.get_aggregated_metrics()
                paper_summary = analyzer.get_paper_metrics_summary()
                
                benchmark_results['gramspec_moe_analysis'] = {
                    'aggregated_metrics': aggregated,
                    'paper_summary': paper_summary,
                }
                
                if self.torch_callback.log_to_console:
                    self.torch_callback._log_debug(f"  ✓ GramSpec MoE Analysis completed")
            except Exception as e:
                self.torch_callback._log_debug(f"  ✗ GramSpec MoE Analysis failed: {e}")
                import traceback
                if self.torch_callback.debug_logging:
                    self.torch_callback._log_debug(traceback.format_exc())
        
        # 2. GramSpec Semantic Validation
        if GRAMSPEC_VALIDATION_AVAILABLE:
            try:
                self.torch_callback._log_debug("Running GramSpec Semantic Validation benchmark...")
                
                # Layer-wise balance 분석을 위한 데이터 수집
                if hasattr(self.torch_callback, 'layer_expert_usage_counts'):
                    layer_expert_usage = self.torch_callback.layer_expert_usage_counts
                    
                    # num_layers 추정
                    num_layers = len(layer_expert_usage) if layer_expert_usage else 0
                    if num_layers == 0:
                        # 모델에서 직접 추출 시도
                        num_layers = sum(1 for _ in model.named_modules() if 'layer' in str(_).lower() or 'block' in str(_).lower())
                    
                    if num_layers > 0 and self.torch_callback.gramspec_validator is None:
                        # Validator 초기화
                        self.torch_callback.gramspec_validator = GramSpecSemanticValidator(
                            num_layers=num_layers,
                            num_experts=self.torch_callback.num_experts
                        )
                    
                    if self.torch_callback.gramspec_validator is not None:
                        layer_balance = self.torch_callback.gramspec_validator.analyze_layer_wise_balance(
                            layer_expert_usage_counts=layer_expert_usage
                        )
                        benchmark_results['gramspec_semantic_validation'] = {
                            'layer_wise_balance': layer_balance,
                        }
                        
                        if self.torch_callback.log_to_console:
                            self.torch_callback._log_debug(f"  ✓ GramSpec Semantic Validation completed")
            except Exception as e:
                self.torch_callback._log_debug(f"  ✗ GramSpec Semantic Validation failed: {e}")
                import traceback
                if self.torch_callback.debug_logging:
                    self.torch_callback._log_debug(traceback.format_exc())
        
        # 3. Expert Specialization Analysis
        if EXPERT_SPECIALIZATION_AVAILABLE and eval_dataloader is not None:
            try:
                self.torch_callback._log_debug("Running Expert Specialization Analysis benchmark...")
                
                # 샘플 데이터 수집
                dataset_samples = []
                max_samples = getattr(args, 'max_eval_samples', 100)
                num_collected = 0
                
                model.eval()
                with torch.no_grad():
                    for batch in eval_dataloader:
                        if num_collected >= max_samples:
                            break
                        
                        # 텍스트 데이터 추출
                        if 'input_ids' in batch and tokenizer is not None:
                            input_ids = batch['input_ids']
                            for i in range(input_ids.shape[0]):
                                if num_collected >= max_samples:
                                    break
                                try:
                                    text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                                    if text.strip():
                                        dataset_samples.append(text)
                                        num_collected += 1
                                except:
                                    continue
                
                if dataset_samples:
                    # Expert activations 수집
                    expert_activations = collect_expert_activations(
                        model=model,
                        tokenizer=tokenizer,
                        dataset=dataset_samples,
                        device=next(model.parameters()).device.type if next(model.parameters()).is_cuda else "cpu",
                        max_samples=min(len(dataset_samples), 100)  # 최대 100개 샘플
                    )
                    
                    # Similarity 계산
                    similarity_matrix = compute_expert_similarity(expert_activations)
                    
                    benchmark_results['expert_specialization'] = {
                        'expert_activations_count': {str(k): len(v) for k, v in expert_activations.items()},
                        'similarity_matrix': similarity_matrix.tolist() if len(similarity_matrix) > 0 else [],
                    }
                    
                    if self.torch_callback.log_to_console:
                        self.torch_callback._log_debug(f"  ✓ Expert Specialization Analysis completed ({len(dataset_samples)} samples)")
            except Exception as e:
                self.torch_callback._log_debug(f"  ✗ Expert Specialization Analysis failed: {e}")
                import traceback
                if self.torch_callback.debug_logging:
                    self.torch_callback._log_debug(traceback.format_exc())
        
        # 4. GramSpec Validation (Perplexity 등)
        if GRAMSPEC_VALIDATION_SCRIPT_AVAILABLE and eval_dataloader is not None:
            try:
                self.torch_callback._log_debug("Running GramSpec Validation benchmark...")
                
                # 샘플 데이터 수집
                eval_dataset = []
                max_samples = getattr(args, 'max_eval_samples', 100)
                num_collected = 0
                
                model.eval()
                with torch.no_grad():
                    for batch in eval_dataloader:
                        if num_collected >= max_samples:
                            break
                        
                        if 'input_ids' in batch and tokenizer is not None:
                            input_ids = batch['input_ids']
                            for i in range(input_ids.shape[0]):
                                if num_collected >= max_samples:
                                    break
                                try:
                                    text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                                    if text.strip():
                                        eval_dataset.append(text)
                                        num_collected += 1
                                except:
                                    continue
                
                if eval_dataset:
                    # Perplexity 평가
                    device = next(model.parameters()).device
                    device_str = device.type if device.is_cuda else "cpu"
                    try:
                        perplexity_results = evaluate_model_perplexity(
                            model=model,
                            tokenizer=tokenizer,
                            eval_dataset=eval_dataset,
                            device=device_str,
                            max_samples=len(eval_dataset)
                        )
                    except Exception as e:
                        self.torch_callback._log_debug(f"  ⚠️ Perplexity evaluation error: {e}")
                        perplexity_results = {'perplexity': 0.0, 'loss': 0.0}
                    
                    benchmark_results['gramspec_validation'] = {
                        'perplexity': perplexity_results,
                    }
                    
                    # Trainer에 로깅
                    if 'trainer' in kwargs:
                        trainer = kwargs['trainer']
                        if hasattr(trainer, 'log'):
                            trainer.log({
                                'eval/benchmark/perplexity': perplexity_results.get('perplexity', 0.0),
                                'eval/benchmark/loss': perplexity_results.get('loss', 0.0),
                            })
                    
                    if self.torch_callback.log_to_console:
                        self.torch_callback._log_debug(f"  ✓ GramSpec Validation completed (PPL: {perplexity_results.get('perplexity', 0.0):.4f})")
            except Exception as e:
                self.torch_callback._log_debug(f"  ✗ GramSpec Validation failed: {e}")
                import traceback
                if self.torch_callback.debug_logging:
                    self.torch_callback._log_debug(traceback.format_exc())
        
        # 5. Efficiency Measurement
        if EFFICIENCY_MEASUREMENT_AVAILABLE:
            try:
                self.torch_callback._log_debug("Running Efficiency Measurement benchmark...")
                
                device = next(model.parameters()).device.type if next(model.parameters()).is_cuda else "cpu"
                input_text = "The capital of France is"  # 기본 입력 텍스트
                
                # Forward throughput 측정
                forward_results = measure_forward_throughput(
                    model=model,
                    tokenizer=tokenizer,
                    input_text=input_text,
                    batch_sizes=[1, 4, 8],
                    seq_length=512,
                    num_runs=20,  # 빠른 측정을 위해 줄임
                    warmup_runs=5,
                    device=device,
                )
                
                # Generation latency 측정
                generation_results = measure_generation_latency(
                    model=model,
                    tokenizer=tokenizer,
                    input_text=input_text,
                    max_new_tokens=32,
                    num_runs=20,
                    warmup_runs=5,
                    device=device,
                )
                
                # FLOPs 추정 (선택적)
                flops_results = {}
                try:
                    flops_results = estimate_flops(
                        model=model,
                        input_shape=(1, 512),
                        device=device,
                    )
                except:
                    pass
                
                benchmark_results['efficiency'] = {
                    'forward_throughput': forward_results,
                    'generation_latency': generation_results,
                    'flops': flops_results,
                }
                
                # Trainer에 로깅
                if 'trainer' in kwargs:
                    trainer = kwargs['trainer']
                    if hasattr(trainer, 'log'):
                        # 주요 지표만 로깅
                        if 1 in forward_results:
                            trainer.log({
                                'eval/benchmark/tokens_per_sec': forward_results[1].get('tokens_per_sec', 0.0),
                                'eval/benchmark/latency_ms': forward_results[1].get('latency_ms_mean', 0.0),
                                'eval/benchmark/gen_latency_ms': generation_results.get('per_token_latency_ms_mean', 0.0),
                            })
                
                if self.torch_callback.log_to_console:
                    if 1 in forward_results:
                        self.torch_callback._log_debug(f"  ✓ Efficiency Measurement completed ({forward_results[1].get('tokens_per_sec', 0.0):.2f} tokens/s)")
            except Exception as e:
                self.torch_callback._log_debug(f"  ✗ Efficiency Measurement failed: {e}")
                import traceback
                if self.torch_callback.debug_logging:
                    self.torch_callback._log_debug(traceback.format_exc())
        
        # 벤치마크 결과 저장
        if benchmark_results and self.torch_callback.save_detailed_logs:
            benchmark_file = os.path.join(
                output_dir,
                f"benchmark_results_step_{state.global_step}.json"
            )
            with open(benchmark_file, 'w') as f:
                json.dump(benchmark_results, f, indent=2)
            self.torch_callback._log_debug(f"Benchmark results saved to {benchmark_file}")
        
        if self.torch_callback.log_to_console:
            self.torch_callback._log_debug(f"Completed {len(benchmark_results)} benchmark(s)")

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
            
            # 벤치마크 실행
            if model is not None and tokenizer is not None:
                self._run_benchmarks(
                    model=model,
                    tokenizer=tokenizer,
                    eval_dataloader=dataloader,
                    state=state,
                    args=args,
                    **kwargs
                )
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
    log_tsne_every: int = 5000,
    tsne_sample_size: int = 2000,
    tokenizer=None,  # ✅ tokenizer를 직접 전달할 수 있도록 추가
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
        log_tsne_every=log_tsne_every,
        tsne_sample_size=tsne_sample_size,
        force_all_ranks=True,  # 모든 프로세스에서 실행 (이미 is_main_process 체크 제거됨)
        **kwargs
    )
    
    # ✅ tokenizer가 전달되면 미리 설정 (VLM 테스트를 위해 필수)
    if tokenizer is not None:
        torch_callback.set_tokenizer(tokenizer)

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
        force_all_ranks=False,  # Multi-GPU 환경에서 rank 0에서만 실행
        **kwargs
    )

    return callback.register_model(model, tokenizer)
