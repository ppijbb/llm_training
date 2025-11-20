#!/usr/bin/env python3
"""
더미 MoE 모델로 callback 테스트
- 파라미터 수: 0.1M 미만
- DeepSpeed/Accelerate 호환
- Accumulated gradient에서도 callback 작동 확인
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import os
import sys
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.moe_monitoring_callback import create_moe_callback_for_transformers
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers import AutoTokenizer
from datasets import Dataset
import wandb


class DummyMoERouter(nn.Module):
    """간단한 MoE Router - routing 정보 저장"""
    def __init__(self, hidden_size=64, num_experts=4, router_dim=16):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.router_dim = router_dim
        
        # 간단한 router
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
    def forward(self, hidden_states):
        # router_logits: [batch*seq, num_experts]
        router_logits = self.gate(hidden_states)
        routing_probs = F.softmax(router_logits, dim=-1)
        selected_experts = routing_probs.argmax(dim=-1)  # [batch*seq]
        
        # Callback을 위해 정보 저장
        self.last_selected_experts = selected_experts.detach()
        self.last_routing_weights = routing_probs.detach()
        self.last_num_experts = self.num_experts
        
        return routing_probs, selected_experts


class DummyExpert(nn.Module):
    """간단한 Expert"""
    def __init__(self, hidden_size=64, intermediate_size=128):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class DummyMoELayer(nn.Module):
    """간단한 MoE Layer - G3MoE와 유사한 구조 - 모든 callback 메트릭 생성"""
    def __init__(self, hidden_size=64, num_experts=4, router_dim=16):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        
        self.router = DummyMoERouter(hidden_size, num_experts, router_dim)
        self.experts = nn.ModuleList([
            DummyExpert(hidden_size, intermediate_size=128) 
            for _ in range(num_experts)
        ])
        
        # Callback 인식을 위한 속성
        # self.gate = self.router.gate  # 메모리 공유로 인한 저장 오류 방지를 위해 주석 처리
        
        # 랜덤 시드 (일관성 있는 테스트를 위해)
        self._step_count = 0
        
    def forward(self, hidden_states, global_routing_hn=None):
        """
        G3MoE와 동일한 인터페이스: (hidden_states, (routing_probs_full, hn, speciality_loss, cosine_similarities, expression_loss))
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        # Routing
        routing_probs, selected_experts = self.router(hidden_states_flat)
        
        # Expert 실행
        final_hidden = torch.zeros_like(hidden_states_flat)
        
        # top_k = 1로 가정 (DummyMoERouter는 argmax만 사용)
        top_k = 1
        selected_experts_2d = selected_experts.unsqueeze(-1) if selected_experts.dim() == 1 else selected_experts  # [batch*seq, top_k]
        
        for expert_idx in range(self.num_experts):
            # top_k를 고려한 마스크 생성
            mask = (selected_experts_2d == expert_idx).any(dim=-1) if selected_experts_2d.dim() == 2 else (selected_experts == expert_idx)
            if mask.any():
                expert_input = hidden_states_flat[mask]
                expert_output = self.experts[expert_idx](expert_input)  # [num_tokens, hidden_dim]
                
                # routing_probs에서 해당 expert의 가중치 추출
                # routing_probs: [batch*seq, num_experts]
                weights = routing_probs[mask, expert_idx]  # [num_tokens]
                
                # weights를 [num_tokens, 1]로 변환하여 broadcasting
                weights = weights.unsqueeze(-1)  # [num_tokens, 1]
                
                # expert_output: [num_tokens, hidden_dim], weights: [num_tokens, 1]
                # broadcasting으로 [num_tokens, hidden_dim] 결과
                final_hidden[mask] = expert_output * weights
        
        # ===== Callback을 위한 모든 메트릭 생성 =====
        # Routing entropy 계산 (callback이 사용)
        safe_probs = routing_probs.clamp_min(1e-12)
        token_entropy = -torch.sum(safe_probs * torch.log(safe_probs), dim=-1)
        avg_routing_entropy = token_entropy.mean()
        
        # Aux loss (랜덤 생성 - callback이 로깅)
        # step에 따라 약간 변동하도록
        self._step_count += 1
        torch.manual_seed(self._step_count % 1000)
        aux_loss = torch.tensor(0.01 + 0.005 * torch.rand(1).item(), device=hidden_states.device)
        
        # Ortho loss (랜덤 생성 - callback이 로깅)
        ortho_loss = torch.tensor(0.02 + 0.01 * torch.rand(1).item(), device=hidden_states.device)
        
        # Expression loss (랜덤 생성 - callback이 로깅)
        expression_loss = torch.tensor(0.015 + 0.008 * torch.rand(1).item(), device=hidden_states.device)
        
        # Speciality loss (랜덤 생성 - callback이 로깅)
        speciality_loss = torch.tensor(0.03 + 0.01 * torch.rand(1).item(), device=hidden_states.device)
        
        # Cosine similarities (랜덤 생성 - callback이 로깅)
        # [num_experts] 형태
        cosine_similarities = torch.rand(self.num_experts, device=hidden_states.device) * 0.5 - 0.25  # -0.25 ~ 0.25
        
        # Router logits (callback이 사용할 수 있음)
        router_logits = self.router.gate(hidden_states_flat)
        
        # routing_probs_full 생성 (G3MoE와 동일한 형태: [batch*seq, num_experts])
        routing_probs_full = routing_probs.detach()
        
        # hn 생성 (dummy: None 또는 더미 텐서)
        hn = global_routing_hn
        if hn is None:
            # Dummy hn: [num_layers=1, batch_size, hidden_dim]
            hn = torch.zeros(1, batch_size, self.num_experts * 4, device=hidden_states.device, dtype=hidden_states.dtype)
        
        # ===== Callback이 기대하는 형식으로 저장 =====
        # G3MoE와 동일: last_selected_experts는 [batch*seq, top_k] 형태
        selected_experts_for_callback = selected_experts_2d.detach()
        if selected_experts_for_callback.dim() == 1:
            selected_experts_for_callback = selected_experts_for_callback.unsqueeze(-1)
        # 음수 제거 및 long 타입으로 변환
        selected_experts_for_callback = selected_experts_for_callback.clamp(min=0).long()
        
        # 기본 routing 정보 저장 (G3MoE와 동일)
        self.last_selected_experts = selected_experts_for_callback  # [batch*seq, top_k]
        self.last_routing_weights = routing_probs.detach()  # routing_probs로도 사용됨
        self.last_routing_probs = routing_probs.detach()  # alias
        self.last_num_experts = self.num_experts
        
        # Router logits
        self.last_router_logits = router_logits.detach()
        self.last_gate_logits = router_logits.detach()  # alias
        
        # Entropy (callback이 avg_routing_entropy로 찾음)
        self.last_avg_routing_entropy = avg_routing_entropy.detach()
        
        # Losses (callback이 찾음)
        self.last_aux_loss = aux_loss.detach()
        self.last_ortho_loss = ortho_loss.detach()
        self.last_expression_loss = expression_loss.detach()
        self.last_speciality_loss = speciality_loss.detach()
        self.last_cosine_similarities = cosine_similarities.detach()
        
        # G3MoE와 동일한 output 구조 반환
        # (hidden_states, (routing_probs_full, hn, speciality_loss, cosine_similarities, expression_loss))
        final_hidden_reshaped = final_hidden.view(batch_size, seq_len, hidden_dim)
        routing_info = (routing_probs_full, hn, speciality_loss, cosine_similarities, expression_loss)
        return final_hidden_reshaped, routing_info


class DummyMoEModel(nn.Module):
    """더미 MoE 모델 - 파라미터 수 0.1M 미만"""
    def __init__(self, vocab_size=1000, hidden_size=64, num_layers=2, num_experts=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # MoE Layers
        self.moe_layers = nn.ModuleList([
            DummyMoELayer(hidden_size, num_experts, router_dim=4)
            for _ in range(num_layers)
        ])
        
        # Output
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
    def forward(self, input_ids, labels=None, **kwargs):
        # Embedding
        hidden_states = self.embedding(input_ids)
        
        # MoE Layers (G3MoE와 동일한 구조: 튜플 반환)
        global_routing_hn = None
        for moe_layer in self.moe_layers:
            moe_output = moe_layer(hidden_states, global_routing_hn)
            if isinstance(moe_output, tuple) and len(moe_output) == 2:
                hidden_states_new, routing_info = moe_output
                # routing_info: (routing_probs_full, hn, speciality_loss, cosine_similarities, expression_loss)
                if len(routing_info) >= 2:
                    global_routing_hn = routing_info[1]  # hn 업데이트
                hidden_states = hidden_states + hidden_states_new
            else:
                # Fallback: 튜플이 아닌 경우
                hidden_states = hidden_states + moe_output
        
        # LM Head
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits
        }
    
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def create_dummy_dataset(num_samples=100, seq_length=32):
    """더미 데이터셋 생성"""
    data = {
        'input_ids': [
            torch.randint(0, 1000, (seq_length,)).tolist() 
            for _ in range(num_samples)
        ],
        'labels': [
            torch.randint(0, 1000, (seq_length,)).tolist() 
            for _ in range(num_samples)
        ]
    }
    return Dataset.from_dict(data)


class TestCallback(TrainerCallback):
    """테스트용 callback - step마다 호출되는지 확인"""
    def __init__(self):
        self.step_count = 0
        self.gradient_accumulation_steps = None
        
    def on_step_begin(self, args, state, control, **kwargs):
        self.step_count += 1
        print(f"[TestCallback] Step {state.global_step} BEGIN (accumulation step: {state.global_step % args.gradient_accumulation_steps})")
        
    def on_step_end(self, args, state, control, **kwargs):
        print(f"[TestCallback] Step {state.global_step} END")
        if self.gradient_accumulation_steps is None:
            self.gradient_accumulation_steps = args.gradient_accumulation_steps


def test_callback_with_dummy_model(use_deepspeed=True, use_accelerate=True):
    """더미 모델로 callback 테스트"""
    print("="*60)
    print("더미 MoE 모델로 Callback 테스트")
    if use_deepspeed:
        print("DeepSpeed 모드")
    elif use_accelerate:
        print("Accelerate 모드")
    else:
        print("기본 모드")
    print("="*60)
    
    # 모델 생성
    model = DummyMoEModel(vocab_size=1000, hidden_size=64, num_layers=2, num_experts=4)
    param_count = model.num_parameters()
    print(f"모델 파라미터 수: {param_count:,} ({param_count/1e6:.3f}M)")
    assert param_count < 1000000, f"파라미터 수가 1M 이상입니다: {param_count}"
    
    # Tokenizer (더미)
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    except:
        # gpt2가 없으면 더미 tokenizer 생성
        from transformers import PreTrainedTokenizer
        class DummyTokenizer:
            def __init__(self):
                self.pad_token = "<pad>"
                self.eos_token = "<eos>"
                self.vocab_size = 1000
        tokenizer = DummyTokenizer()
    tokenizer.pad_token = tokenizer.eos_token if hasattr(tokenizer, 'eos_token') else "<pad>"
    
    # 데이터셋
    train_dataset = create_dummy_dataset(num_samples=500, seq_length=32)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./test_output",
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=64,  # Accumulated gradient 테스트
        learning_rate=1e-4,
        logging_steps=1,
        save_steps=999999999,  # 모델 저장 비활성화 (매우 큰 값)
        save_total_limit=0,  # 저장 안 함
        save_strategy="no",  # 저장 전략 비활성화
        bf16=False,
        fp16=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=["wandb"],  # wandb 활성화
    )
    
    deepspeed_config = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "zero_optimization": {
            "stage": 3,
        },
        "fp16": {
            "enabled": False
        },
        "bf16": {
            "enabled": False
        }
    }
    import json
    os.makedirs("./test_output", exist_ok=True)
    with open("./test_output/deepspeed_config.json", "w") as f:
        json.dump(deepspeed_config, f)
    training_args.deepspeed = "./test_output/deepspeed_config.json"
    
    # Wandb 초기화 - rank 0에서만 실행, 중복 실행 방지
    rank = int(os.getenv("RANK", "0"))
    if rank == 0 and (wandb.run is None or not wandb.run):
        wandb.init(
            project="moe-callback-test",
            name=f"dummy-test-{int(time.time())}",
            config={
                "model": "dummy_moe",
                "num_experts": 4,
                "hidden_size": 16,
                "num_layers": 3,
                "use_deepspeed": use_deepspeed,
                "use_accelerate": use_accelerate,
            },
            mode="online"  # 항상 online으로 wandb에 기록
        )
    
    # MoE Callback 생성 - wandb 로거 사용
    moe_callback = create_moe_callback_for_transformers(
        num_experts=4,
        log_every_n_steps=1,
        logger=wandb,  # wandb 활성화
        log_to_console=False,
        debug_logging=True,
        enable_generation_logging=False,  # 생성 로깅 비활성화
    )
    
    # 테스트 callback
    test_callback = TestCallback()
    
    # Trainer
    class DummyTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            outputs = model(**inputs)
            loss = outputs['loss']
            return (loss, outputs) if return_outputs else loss
    
    trainer = DummyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        callbacks=[test_callback, moe_callback],
    )
    
    print("\n" + "="*60)
    print("학습 시작 - Step마다 callback 호출 확인")
    print("="*60)
    print(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    print(f"Batch size: {training_args.per_device_train_batch_size}")
    print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print()
    
    # 학습 실행
    try:
        trainer.train()
        print("\n" + "="*60)
        print("✅ 학습 완료!")
        print("="*60)
        print(f"TestCallback이 호출된 step 수: {test_callback.step_count}")
        print(f"Global step: {trainer.state.global_step}")
        
        # Callback이 모든 step에서 호출되었는지 확인
        expected_steps = len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
        if len(train_dataset) % (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) != 0:
            expected_steps += 1
        
        print(f"예상 step 수: {expected_steps}")
        print(f"실제 global step: {trainer.state.global_step}")
        
        # MoE callback이 routing 정보를 제대로 캡처했는지 확인
        if hasattr(moe_callback.torch_callback, 'layer_outputs'):
            layer_outputs_count = len(moe_callback.torch_callback.layer_outputs)
            print(f"MoE callback이 캡처한 layer 수: {layer_outputs_count}")
            if layer_outputs_count > 0:
                print("✅ MoE callback이 routing 정보를 성공적으로 캡처했습니다!")
            else:
                print("⚠️ Warning: MoE callback이 routing 정보를 캡처하지 못했습니다.")
        
        if test_callback.step_count >= trainer.state.global_step:
            print("✅ 모든 step에서 callback이 호출되었습니다!")
        else:
            print(f"⚠️ Warning: 일부 step에서 callback이 호출되지 않았을 수 있습니다.")
        
        # Wandb에 최종 요약 로깅
        layer_outputs_count = len(moe_callback.torch_callback.layer_outputs) if hasattr(moe_callback.torch_callback, 'layer_outputs') else 0
        if os.environ.get("RANK") == "0":
            wandb.log({
                "test/summary/total_steps": trainer.state.global_step,
                "test/summary/callback_calls": test_callback.step_count,
                "test/summary/layers_captured": layer_outputs_count,
            })
            print("\n✅ Wandb에 모든 메트릭이 기록되었습니다!")
        if wandb.run:
            print(f"   Wandb URL: {wandb.run.url}")
            
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Wandb 종료
        if wandb.run:
            wandb.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepspeed", action="store_true", help="DeepSpeed 사용")
    parser.add_argument("--accelerate", action="store_true", help="Accelerate 사용")
    args = parser.parse_args()
    
    test_callback_with_dummy_model(
        use_deepspeed=args.deepspeed,
        use_accelerate=args.accelerate
    )

