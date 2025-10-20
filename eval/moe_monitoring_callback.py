import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict, deque
from typing import Dict, Any, Optional, Callable
import json
import time
import os

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
    """Pure PyTorch MoE monitoring callback"""
    
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
        debug_logging: bool = False       
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
        
        # 내부 상태 (step 제거)
        self.expert_usage_history = defaultdict(lambda: deque(maxlen=window_size))
        self.routing_stats = defaultdict(list)
        self.alerts_history = []
        self.detailed_logs = []
        
        # hooks 저장소
        self.hooks = []
        self.layer_outputs = {}
        
        if save_detailed_logs:
            import os
            os.makedirs(log_dir, exist_ok=True)
    
    def _log_debug(self, message: str):
        """내부 디버그 메시지 로깅"""
        if self.debug_logging and self.log_to_console:
            print(f"[MoE Debug] {message}")
    
    def register_model(self, model: torch.nn.Module):
        """모델에 hooks 등록"""
        self.model = model
        self._register_hooks()
        return self
    
    def _register_hooks(self):
        """MoE 레이어에 forward hooks 등록"""
        for name, module in self.model.named_modules():
            if self._is_moe_layer(module):
                hook = module.register_forward_hook(
                    self._create_hook_fn(name)
                )
                self.hooks.append(hook)
    
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
                    self._log_debug(f"{layer_name}: extracted {list(routing_info.keys())}")
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

        # 실제 G3MoE/GRIN 모델 구조에 맞춘 추출
        # output이 (hidden_states, router_logits) 튜플인 경우
        if isinstance(output, tuple) and len(output) == 2:
            hidden_states, router_logits = output
            if router_logits is not None:
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
        # 1. 속성으로 저장된 경우
        for attr in ['last_expert_assignments', 'expert_assignments', 'selected_experts']:
            if hasattr(module, attr):
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

        return routing_info if routing_info else None
    
    def on_step_begin(self):
        """Step 시작 시 호출"""
        self.layer_outputs.clear()
    
    def on_step_end(self, current_step: int, **kwargs):
        """Step 종료 시 호출 - current_step은 필수 매개변수"""
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


    @torch.no_grad()
    def _calculate_step_metrics(self):
        """현재 step의 메트릭 계산"""
        metrics = {}
        
        for layer_name, routing_info in self.layer_outputs.items():
            layer_metrics = {}
            
            expert_assignments = routing_info.get('expert_assignments')
            routing_probs = routing_info.get('routing_probs')
            # 일관된 num_experts 사용 (스텝마다 변하지 않도록)
            num_experts = self.num_experts
            
            if expert_assignments is not None:
                if expert_assignments.is_cuda: # GPU tensor인 경우에만
                    expert_assignments = expert_assignments.clamp(0, num_experts - 1)
                
                # Expert 사용 분포
                if expert_assignments.dim() > 1:
                    expert_assignments = expert_assignments.flatten()
                
                usage_counts = torch.bincount(expert_assignments, minlength=num_experts)
                self.expert_usage_history[layer_name].append(usage_counts.cpu())
                
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
        
        return metrics
    
    def _log_metrics(self, metrics, current_step: int):
        """메트릭 로깅"""
        # rank 0에서만 로깅 수행
        if not self.is_main_process:
            return
            
        log_data = {}
        
        # 레이어별 메트릭
        for layer_name, layer_metrics in metrics.items():
            for metric_name, value in layer_metrics.items():
                if torch.is_tensor(value) and value.numel() == 1:
                    log_data[f'moe/{layer_name}/{metric_name}'] = value.item()
                elif isinstance(value, (int, float)):
                    log_data[f'moe/{layer_name}/{metric_name}'] = value
        
        # 전체 평균 메트릭
        if metrics:
            avg_cv = np.mean([m.get('expert_cv', torch.tensor(0.0)).item() for m in metrics.values() if 'expert_cv' in m])
            avg_entropy = np.mean([m.get('routing_entropy', torch.tensor(0.0)).item() for m in metrics.values() if 'routing_entropy' in m])
            total_unused = sum([m.get('unused_experts', 0) for m in metrics.values()])
            
            log_data.update({
                'moe/avg_expert_cv': avg_cv,
                'moe/avg_routing_entropy': avg_entropy,
                'moe/total_unused_experts': total_unused,
                'moe/step': current_step
            })
        
        # 로거에 전송 (rank 0에서만)
        if self.logger:
            if hasattr(self.logger, 'log'):
                # Let the logger infer step from the training loop to avoid mismatches
                self.logger.log(log_data)
            elif hasattr(self.logger, 'add_scalars'):  # TensorBoard
                for key, value in log_data.items():
                    self.logger.add_scalar(key, value, current_step)
        
        # 콘솔 출력
        if self.log_to_console:
            self._log_debug(f"Step {current_step} MoE Metrics:")
            for key, value in log_data.items():
                if 'avg_' in key or 'total_' in key:
                    self._log_debug(f"  {key}: {value:.4f}")
    
    def _log_heatmaps(self, current_step: int):
        """Expert 사용률 히트맵 로깅"""
        # rank 0에서만 실행
        if not self.is_main_process:
            return
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            for layer_name in self.expert_usage_history:
                if len(self.expert_usage_history[layer_name]) > 10:
                    # 모든 텐서를 동일한 크기로 맞추기 위해 최대 크기로 패딩
                    usage_tensors = list(self.expert_usage_history[layer_name])
                    max_size = max(tensor.size(0) for tensor in usage_tensors)
                    
                    # 각 텐서를 최대 크기로 패딩
                    padded_tensors = []
                    for tensor in usage_tensors:
                        if tensor.size(0) < max_size:
                            padding = torch.zeros(max_size - tensor.size(0), dtype=tensor.dtype)
                            padded_tensor = torch.cat([tensor, padding])
                        else:
                            padded_tensor = tensor
                        padded_tensors.append(padded_tensor)
                    
                    usage_matrix = torch.stack(padded_tensors)
                    usage_matrix = usage_matrix.float()
                    
                    # 정규화
                    row_sums = usage_matrix.sum(dim=1, keepdim=True)
                    usage_matrix = usage_matrix / (row_sums + 1e-8)
                    
                    plt.figure(figsize=(12, 6))
                    sns.heatmap(usage_matrix.T.numpy(), 
                                cmap='YlOrRd', 
                                xticklabels=False,
                                yticklabels=True)
                    plt.title(f'{layer_name} Expert Usage Distribution')
                    plt.xlabel('Time Steps')
                    plt.ylabel('Expert Index')
                    
                    if self.logger and hasattr(self.logger, 'log'):
                        import wandb
                        self.logger.log({
                            f'moe/{layer_name}/usage_heatmap': wandb.Image(plt)
                        }, step=current_step)
                    
                    if self.save_detailed_logs:
                        plt.savefig(f'{self.log_dir}/{layer_name}_heatmap_step_{current_step}.png')
                    
                    plt.close()
        except ImportError:
            self._log_debug("matplotlib/seaborn not available for heatmap logging")
    
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
            if 'unused_experts' in layer_metrics:
                unused = layer_metrics['unused_experts']
                total_experts = layer_metrics.get('usage_counts', torch.zeros(self.num_experts)).numel()
                if unused / total_experts > self.unused_expert_threshold:
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
            
            if self.logger and hasattr(self.logger, 'log'):
                self.logger.log({
                    f'alerts/{alert["type"]}': 1,
                    f'alerts/{alert["layer"]}_severity': alert.get('severity', 1)
                }, step=current_step)
    
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
        **kwargs
    ):
        """훈련 시작 시 모델 등록"""
        if model is not None and not self._model_registered:
            self.torch_callback.register_model(model)
            self._model_registered = True
            self.torch_callback._log_debug(f"MoE monitoring registered for model with {len(self.torch_callback.hooks)} MoE layers")
    
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
    **kwargs
) -> TransformersMoECallbackWrapper:
    """Transformers용 MoE 콜백 생성 편의 함수"""
    
    torch_callback = TorchMoECallback(
        log_every_n_steps=log_every_n_steps,
        logger=logger,
        **kwargs
    )
    
    return TransformersMoECallbackWrapper(torch_callback)

def create_moe_callback_for_pytorch(
    model: torch.nn.Module,
    log_every_n_steps: int = 100,
    logger=None,
    **kwargs
) -> TorchMoECallback:
    """순수 PyTorch용 MoE 콜백 생성 편의 함수"""
    
    callback = TorchMoECallback(
        log_every_n_steps=log_every_n_steps,
        logger=logger,
        **kwargs
    )
    
    return callback.register_model(model)
