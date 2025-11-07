# coding=utf-8
"""
Checkpoint 모델 평가 스크립트

학습된 checkpoint 모델을 불러와서 GramSpec MoE 분석을 수행합니다.
- Load balancing metrics
- Expert specialization metrics
- Routing quality metrics
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import json
from tqdm import tqdm
import argparse
from peft import PeftModel
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from transformers.generation.configuration_utils import GenerationConfig
from transformers.image_utils import load_image
import copy

from models import G3MoEModel, G3MoETextModel, G3MoEConfig, G3MoEForCausalLM, G3MoEForConditionalGeneration, G3MoETextConfig
from transformers.modeling_utils import VLMS
from eval.gramspec_moe_analysis import GramSpecAnalyzer

# Register models
AutoConfig.register("g3moe", G3MoEConfig)
AutoConfig.register("g3moe_text", G3MoETextConfig)
AutoModel.register(G3MoEConfig, G3MoEModel)
AutoModel.register(G3MoETextConfig, G3MoETextModel)
AutoModelForCausalLM.register(G3MoETextConfig, G3MoEForCausalLM)
VLMS.append("g3moe")


class RoutingInfoCollector:
    """모델 forward pass에서 routing 정보를 수집하는 hook"""
    
    def __init__(self, analyzer: GramSpecAnalyzer):
        self.analyzer = analyzer
        self.hooks = []
        self.router_hooks = []
        self.routing_data = []
        self.router_internal_data = defaultdict(list)
        
    def register_hooks(self, model: nn.Module):
        """모델의 MoE 레이어와 Router에 hook 등록"""
        # Router의 forward hook (routing_logits, expression_logits 추출)
        def create_router_hook(layer_name):
            def router_hook_fn(module, input, output):
                # Router forward의 반환값: (multiplier, selected_experts, expression_logits, hn, speciality_penalty, cosine_similarities, expression_loss)
                if len(output) >= 7:
                    multiplier, selected_experts, expression_logits, hn, speciality_penalty, cosine_similarities, expression_loss = output
                    
                    # Router 내부에서 routing_logits 추출 시도
                    routing_logits = None
                    if hasattr(module, 'load_balancer'):
                        # GRU의 출력을 routing_logits로 사용
                        # input[0]은 hidden_states, input[1]은 hn
                        if len(input) >= 1:
                            hidden_states = input[0]
                            hn_input = input[1] if len(input) > 1 else None
                            
                            # GRU forward를 직접 호출하여 routing_logits 얻기
                            with torch.no_grad():
                                if hn_input is not None:
                                    routing_logits, _ = module.load_balancer(hidden_states, hn_input.to(hidden_states.dtype))
                                else:
                                    routing_logits, _ = module.load_balancer(hidden_states, None)
                                
                                # Reshape to [batch, seq, num_experts, router_dim]
                                batch_size, seq_len = hidden_states.shape[:2]
                                num_experts = module.num_experts
                                router_dim = module.router_dim
                                routing_logits = routing_logits.view(batch_size, seq_len, num_experts, router_dim)
                    
                    self.router_internal_data[layer_name].append({
                        'routing_logits': routing_logits.detach().cpu() if routing_logits is not None else None,
                        'expression_logits': expression_logits.detach().cpu() if isinstance(expression_logits, torch.Tensor) else None,
                        'selected_experts': selected_experts.detach().cpu(),
                        'routing_weights': multiplier.detach().cpu(),
                        'cosine_similarities': cosine_similarities.detach().cpu() if isinstance(cosine_similarities, torch.Tensor) else None,
                        'speciality_penalty': float(speciality_penalty) if isinstance(speciality_penalty, torch.Tensor) else speciality_penalty,
                        'expression_loss': float(expression_loss) if isinstance(expression_loss, torch.Tensor) else expression_loss,
                    })
            return router_hook_fn
        
        # MoE Block의 forward hook
        def create_moe_hook(layer_name):
            def moe_hook_fn(module, input, output):
                # GramSpecMoEBlock에서 routing 정보 추출
                if hasattr(module, '_last_routing_info'):
                    routing_info = module._last_routing_info
                    if routing_info is not None and len(routing_info) >= 6:
                        routing_weights, hn, speciality_loss, cosine_similarities, expression_loss, router_scores = routing_info
                        
                        # Router에서 수집한 데이터와 매칭
                        router_data = self.router_internal_data.get(layer_name, [])
                        if router_data:
                            latest_router_data = router_data[-1]  # 가장 최근 데이터 사용
                            
                            self.routing_data.append({
                                'layer': layer_name,
                                'routing_logits': latest_router_data.get('routing_logits'),
                                'expression_logits': latest_router_data.get('expression_logits'),
                                'routing_weights': latest_router_data.get('routing_weights'),
                                'selected_experts': latest_router_data.get('selected_experts'),
                                'cosine_similarities': latest_router_data.get('cosine_similarities'),
                                'speciality_penalty': latest_router_data.get('speciality_penalty', float(speciality_loss) if isinstance(speciality_loss, torch.Tensor) else speciality_loss),
                                'expression_loss': latest_router_data.get('expression_loss', float(expression_loss) if isinstance(expression_loss, torch.Tensor) else expression_loss),
                                'router_scores': router_scores.detach().cpu(),
                            })
            return moe_hook_fn
        
        # Router와 MoE Block에 hook 등록
        for name, module in model.named_modules():
            # Router hook
            if hasattr(module, 'load_balancer') and hasattr(module, 'expression_projector'):
                # GramSpecRouter
                hook = module.register_forward_hook(create_router_hook(name))
                self.router_hooks.append(hook)
                print(f"Registered router hook: {name}")
            
            # MoE Block hook
            if isinstance(module, nn.Module) and hasattr(module, '_last_routing_info'):
                hook = module.register_forward_hook(create_moe_hook(name))
                self.hooks.append(hook)
                print(f"Registered MoE block hook: {name}")
    
    def remove_hooks(self):
        """Hook 제거"""
        for hook in self.hooks + self.router_hooks:
            hook.remove()
        self.hooks = []
        self.router_hooks = []
    
    def analyze_collected_data(self, num_experts: int, router_dim: int = 128) -> Dict[str, Any]:
        """수집된 데이터 분석"""
        if not self.routing_data:
            print("⚠️  No routing data collected. Make sure hooks are registered correctly.")
            return {}
        
        all_metrics = []
        
        for data in self.routing_data:
            routing_logits = data.get('routing_logits')
            expression_logits = data.get('expression_logits')
            selected_experts = data.get('selected_experts')
            routing_weights = data.get('routing_weights')
            cosine_similarities = data.get('cosine_similarities')
            
            if routing_logits is None or expression_logits is None:
                # Router scores로부터 근사값 생성
                router_scores = data.get('router_scores')
                if router_scores is not None:
                    batch_size, seq_len, num_experts_actual = router_scores.shape
                    # Router scores를 기반으로 근사
                    routing_logits = router_scores.unsqueeze(-1).expand(-1, -1, -1, router_dim)
                    expression_logits = routing_logits.clone()
                else:
                    continue
            
            if selected_experts is None or routing_weights is None:
                # router_scores에서 추출
                router_scores = data.get('router_scores')
                if router_scores is not None:
                    batch_size, seq_len, num_experts_actual = router_scores.shape
                    top_k = routing_weights.shape[-1] if routing_weights is not None else 2
                    routing_scores_flat = router_scores.view(batch_size * seq_len, num_experts_actual)
                    top_k_values, selected_experts = torch.topk(routing_scores_flat, k=min(top_k, num_experts_actual), dim=-1)
                    routing_weights = torch.softmax(top_k_values, dim=-1)
                else:
                    continue
            
            # Shape 확인 및 변환
            if routing_logits.dim() == 4:
                batch_size, seq_len, num_experts_actual, router_dim_actual = routing_logits.shape
                routing_logits = routing_logits.view(batch_size * seq_len, num_experts_actual, router_dim_actual)
                expression_logits = expression_logits.view(batch_size * seq_len, num_experts_actual, router_dim_actual)
            
            if cosine_similarities is None:
                batch_size, seq_len = routing_logits.shape[0] // num_experts, 1
                cosine_similarities = torch.zeros(batch_size, seq_len, num_experts)
            
            # Analyzer에 전달
            try:
                metrics = self.analyzer.analyze_routing_step(
                    routing_logits=routing_logits,
                    expression_logits=expression_logits,
                    selected_experts=selected_experts,
                    routing_weights=routing_weights,
                    speciality_penalty=data.get('speciality_penalty', 0.0),
                    cosine_similarities=cosine_similarities,
                    expression_loss=data.get('expression_loss', 0.0),
                )
                
                metrics['layer'] = data['layer']
                all_metrics.append(metrics)
            except Exception as e:
                print(f"Error analyzing data for layer {data.get('layer', 'unknown')}: {e}")
                continue
        
        return {
            'per_layer_metrics': all_metrics,
            'aggregated_metrics': self.analyzer.get_aggregated_metrics(),
            'paper_summary': self.analyzer.get_paper_metrics_summary(),
        }


def load_checkpoint_model(
    checkpoint_path: str,
    base_model_name: str,
    model_architecture,
    moe_config: Dict[str, Any],
    device: str = "cuda",
) -> Tuple[nn.Module, Any]:
    """Checkpoint에서 모델과 토크나이저 로드"""
    print(f"Loading base model: {base_model_name}")
    base_config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
    base_config = base_config.to_dict()
    
    if "text_config" not in base_config:
        base_config['text_config'] = copy.deepcopy(base_config)
    
    base_config['text_config'].update(moe_config)
    base_config.update(base_config['text_config'])
    model_config = G3MoEConfig(**base_config)
    model_config.model_type = "gemma3"
    model_config.text_config.model_type = "gemma3_text"
    model_config.architectures = ["G3MoEForConditionalGeneration"]
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    model = PeftModel.from_pretrained(
        model=model_architecture.from_pretrained(
            pretrained_model_name_or_path=base_model_name,
            config=model_config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            offload_state_dict=True,
            use_cache=False,
            attn_implementation="flash_attention_3",
        ).to(device),
        model_id=checkpoint_path,
    )
    model.merge_and_unload()
    model.eval()
    
    # Tokenizer 로드
    try:
        tokenizer = AutoProcessor.from_pretrained(base_model_name, use_fast=True)
        if hasattr(tokenizer, 'chat_template'):
            chat_template_path = "/home/conan/workspace/llm_training/sft/config/chat_template.txt"
            if os.path.exists(chat_template_path):
                with open(chat_template_path, "r") as f:
                    tokenizer.chat_template = f.read()
    except:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    
    print("✅ Model and tokenizer loaded successfully")
    return model, tokenizer


def prepare_evaluation_data(
    tokenizer: Any,
    dataset_name: Optional[str] = None,
    num_samples: int = 100,
    max_length: int = 512,
    use_training_eval_set: bool = True,
) -> List[Dict[str, torch.Tensor]]:
    """
    평가용 데이터 준비
    
    Options:
    1. 학습에 사용한 test split 사용 (use_training_eval_set=True)
    2. HuggingFace 데이터셋 사용 (dataset_name 지정)
    3. VLM 평가 데이터셋 사용 (MME, VQAv2 등)
    """
    inputs_list = []
    
    # Option 1: 학습에 사용한 test split 사용
    if use_training_eval_set and dataset_name:
        try:
            from data.simple_sft_dataset import get_simple_sft_dataset
            print(f"Loading training eval set from: {dataset_name}")
            dataset = get_simple_sft_dataset(
                dataset_name=dataset_name,
                tokenizer=tokenizer,
                max_length=max_length,
                max_samples=num_samples,
                test_size=0.1,
                use_streaming=False
            )
            
            eval_dataset = dataset.get("test", None)
            if eval_dataset is not None:
                print(f"✅ Loaded {len(eval_dataset)} eval samples from training dataset")
                eval_dataset = eval_dataset.select(range(min(num_samples, len(eval_dataset))))
                
                for sample in tqdm(eval_dataset, desc="Preparing eval data"):
                    # VLM 데이터셋인 경우 이미지 포함
                    if 'images' in sample and sample['images']:
                        # 이미지가 있는 경우
                        messages = sample.get('messages', [])
                        if messages:
                            # Chat template 적용
                            if hasattr(tokenizer, 'apply_chat_template'):
                                text = tokenizer.apply_chat_template(
                                    messages,
                                    add_generation_prompt=True,
                                    tokenize=False
                                )
                            else:
                                text = str(messages)
                            
                            # 이미지와 텍스트 함께 처리
                            images = sample['images']
                            if isinstance(images, list) and len(images) > 0:
                                inputs = tokenizer(
                                    text=text,
                                    images=images[0] if len(images) == 1 else images,
                                    return_tensors="pt",
                                    truncation=True,
                                    max_length=max_length,
                                    padding="max_length",
                                )
                            else:
                                inputs = tokenizer(
                                    text=text,
                                    return_tensors="pt",
                                    truncation=True,
                                    max_length=max_length,
                                    padding="max_length",
                                )
                        else:
                            continue
                    else:
                        # 텍스트만 있는 경우
                        messages = sample.get('messages', [])
                        if messages:
                            if hasattr(tokenizer, 'apply_chat_template'):
                                text = tokenizer.apply_chat_template(
                                    messages,
                                    add_generation_prompt=True,
                                    tokenize=False
                                )
                            else:
                                text = str(messages)
                            
                            inputs = tokenizer(
                                text=text,
                                return_tensors="pt",
                                truncation=True,
                                max_length=max_length,
                                padding="max_length",
                            )
                        else:
                            continue
                    
                    if "token_type_ids" in inputs:
                        del inputs["token_type_ids"]
                    inputs_list.append(inputs)
                
                return inputs_list
        except Exception as e:
            print(f"⚠️ Failed to load training eval set: {e}")
            print("Falling back to default samples...")
    
    # Option 2: VLM 평가 데이터셋 사용
    if dataset_name and dataset_name.lower() in ['mme', 'vqav2', 'textvqa', 'imagenet1k', 'imagenet-1k']:
        try:
            from datasets import load_dataset
            from PIL import Image
            import requests
            from io import BytesIO
            
            print(f"Loading VLM evaluation dataset: {dataset_name}")
            
            if dataset_name.lower() == 'mme':
                dataset = load_dataset("MMMU/MME")
                # MME는 여러 task로 구성
                tasks = ['color', 'count', 'position', 'posters', 'ocr']
                for task in tasks[:2]:  # 처음 2개 task만 사용
                    if task in dataset:
                        task_data = dataset[task].select(range(min(num_samples // 2, len(dataset[task]))))
                        for sample in tqdm(task_data, desc=f"Loading {task}"):
                            image = sample['image']
                            question = sample['question']
                            prompt = f"<image>\n{question}\nAnswer:"
                            
                            inputs = tokenizer(
                                text=[prompt],
                                images=[image],
                                return_tensors="pt",
                                truncation=True,
                                max_length=max_length,
                            )
                            if "token_type_ids" in inputs:
                                del inputs["token_type_ids"]
                            inputs_list.append(inputs)
            
            elif dataset_name.lower() == 'vqav2':
                print("Loading VQAv2 dataset from HuggingFace...")
                try:
                    # VQAv2 데이터셋 로드
                    dataset = load_dataset("lmms-lab/VQAv2", split="validation")
                except:
                    # 대체 경로 시도
                    try:
                        dataset = load_dataset("datalab/vqa_v2", split="validation")
                    except:
                        dataset = load_dataset("Antonio/vqa_v2", split="validation")
                
                dataset = dataset.select(range(min(num_samples, len(dataset))))
                
                for sample in tqdm(dataset, desc="Loading VQAv2"):
                    try:
                        # 이미지 로드
                        if 'image' in sample:
                            image = sample['image']
                        elif 'image_url' in sample:
                            # URL에서 이미지 다운로드
                            img_url = sample['image_url']
                            response = requests.get(img_url, timeout=10)
                            image = Image.open(BytesIO(response.content)).convert('RGB')
                        elif 'image_path' in sample:
                            image = Image.open(sample['image_path']).convert('RGB')
                        else:
                            continue
                        
                        # 질문 추출
                        question = sample.get('question', sample.get('text', ''))
                        if not question:
                            continue
                        
                        # 프롬프트 생성
                        prompt = f"<image>\nQuestion: {question}\nAnswer:"
                        
                        inputs = tokenizer(
                            text=[prompt],
                            images=[image],
                            return_tensors="pt",
                            truncation=True,
                            max_length=max_length,
                        )
                        if "token_type_ids" in inputs:
                            del inputs["token_type_ids"]
                        inputs_list.append(inputs)
                    except Exception as e:
                        print(f"⚠️ Error loading VQAv2 sample: {e}")
                        continue
            
            elif dataset_name.lower() == 'textvqa':
                print("Loading TextVQA dataset from HuggingFace...")
                try:
                    dataset = load_dataset("lmms-lab/TextVQA", split="validation")
                except:
                    try:
                        dataset = load_dataset("textvqa", split="validation")
                    except:
                        dataset = load_dataset("HuggingFaceM4/TextVQA", split="validation")
                
                dataset = dataset.select(range(min(num_samples, len(dataset))))
                
                for sample in tqdm(dataset, desc="Loading TextVQA"):
                    try:
                        # 이미지 로드
                        if 'image' in sample:
                            image = sample['image']
                        elif 'image_url' in sample:
                            img_url = sample['image_url']
                            response = requests.get(img_url, timeout=10)
                            image = Image.open(BytesIO(response.content)).convert('RGB')
                        elif 'image_path' in sample:
                            image = Image.open(sample['image_path']).convert('RGB')
                        else:
                            continue
                        
                        # 질문 추출
                        question = sample.get('question', sample.get('text', ''))
                        if not question:
                            continue
                        
                        # 프롬프트 생성 (TextVQA는 텍스트가 포함된 이미지에 대한 질문)
                        prompt = f"<image>\nQuestion: {question}\nAnswer the question based on the text visible in the image:"
                        
                        inputs = tokenizer(
                            text=[prompt],
                            images=[image],
                            return_tensors="pt",
                            truncation=True,
                            max_length=max_length,
                        )
                        if "token_type_ids" in inputs:
                            del inputs["token_type_ids"]
                        inputs_list.append(inputs)
                    except Exception as e:
                        print(f"⚠️ Error loading TextVQA sample: {e}")
                        continue
            
            elif dataset_name.lower() in ['imagenet1k', 'imagenet-1k']:
                print("Loading ImageNet-1k dataset from HuggingFace...")
                try:
                    # ImageNet-1k 데이터셋 로드
                    dataset = load_dataset("imagenet-1k", split="validation")
                except:
                    try:
                        dataset = load_dataset("Maysee/tiny-imagenet", split="validation")
                        print("⚠️ Using tiny-imagenet as fallback")
                    except:
                        # ImageNet 직접 경로 시도
                        try:
                            dataset = load_dataset("laion/laion400m", split="train", streaming=True)
                            print("⚠️ Using LAION-400M as fallback (will sample first N)")
                            dataset = list(dataset.take(num_samples))
                        except Exception as e:
                            raise Exception(f"Could not load ImageNet-1k: {e}")
                
                # ImageNet은 이미지 분류이므로 클래스 이름을 질문으로 사용
                if not isinstance(dataset, list):
                    dataset = dataset.select(range(min(num_samples, len(dataset))))
                
                for sample in tqdm(dataset, desc="Loading ImageNet-1k"):
                    try:
                        # 이미지 로드
                        if 'image' in sample:
                            image = sample['image']
                        elif 'img' in sample:
                            image = sample['img']
                        else:
                            continue
                        
                        # 레이블 추출
                        label = sample.get('label', sample.get('labels', None))
                        label_text = sample.get('label_text', sample.get('class_name', ''))
                        
                        # 프롬프트 생성 (이미지 분류)
                        if label_text:
                            prompt = f"<image>\nWhat is the main object or class in this image? Answer with a single word or short phrase:"
                        else:
                            prompt = f"<image>\nWhat is the main object or class in this image? Answer with a single word or short phrase:"
                        
                        inputs = tokenizer(
                            text=[prompt],
                            images=[image],
                            return_tensors="pt",
                            truncation=True,
                            max_length=max_length,
                        )
                        if "token_type_ids" in inputs:
                            del inputs["token_type_ids"]
                        
                        # 레이블 정보도 함께 저장 (정확도 계산용)
                        inputs['label'] = label
                        inputs['label_text'] = label_text
                        inputs_list.append(inputs)
                    except Exception as e:
                        print(f"⚠️ Error loading ImageNet-1k sample: {e}")
                        continue
            
            if inputs_list:
                print(f"✅ Successfully loaded {len(inputs_list)} samples from {dataset_name}")
                return inputs_list
        except Exception as e:
            import traceback
            print(f"⚠️ Failed to load VLM dataset {dataset_name}: {e}")
            traceback.print_exc()
            print("Falling back to default samples...")
    
    # Option 3: 기본 샘플 (fallback)
    print("Using default text samples...")
    sample_texts = [
        "The capital of France is Paris.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language.",
        "The Earth orbits around the Sun.",
        "Water boils at 100 degrees Celsius at sea level.",
    ] * (num_samples // 5 + 1)
    
    sample_texts = sample_texts[:num_samples]
    
    for text in tqdm(sample_texts, desc="Preparing evaluation data"):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
        inputs_list.append(inputs)
    
    return inputs_list


def evaluate_model(
    model: nn.Module,
    tokenizer: Any,
    eval_data: List[Dict[str, torch.Tensor]],
    analyzer: GramSpecAnalyzer,
    device: str = "cuda",
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """모델 평가 실행"""
    print(f"\n{'='*60}")
    print("Starting Model Evaluation")
    print(f"{'='*60}")
    
    collector = RoutingInfoCollector(analyzer)
    collector.register_hooks(model)
    
    model.eval()
    eval_data = eval_data[:max_samples] if max_samples else eval_data
    
    print(f"Evaluating on {len(eval_data)} samples...")
    with torch.no_grad():
        for i, inputs in enumerate(tqdm(eval_data, desc="Evaluating")):
            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward pass
            try:
                outputs = model(**inputs)
            except Exception as e:
                print(f"Error in forward pass for sample {i}: {e}")
                continue
    
    # 분석 수행
    print("\nAnalyzing collected routing data...")
    results = collector.analyze_collected_data(
        num_experts=analyzer.num_experts,
        router_dim=analyzer.router_dim,
    )
    
    collector.remove_hooks()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint model with GramSpec analysis")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to checkpoint directory")
    parser.add_argument("--base_model", type=str, default="Gunulhona/Gemma-3-4B",
                       help="Base model name")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of evaluation samples")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--num_experts", type=int, default=8,
                       help="Number of experts (from config)")
    parser.add_argument("--router_dim", type=int, default=128,
                       help="Router dimension")
    parser.add_argument("--eval_dataset", type=str, default=None,
                       help="Evaluation dataset name (e.g., 'HuggingFaceTB/smoltalk' for training eval set, or 'mme', 'vqav2', 'textvqa', 'imagenet1k' for VLM benchmarks)")
    parser.add_argument("--use_training_eval_set", action="store_true",
                       help="Use test split from training dataset")
    
    args = parser.parse_args()
    
    # MoE config (checkpoint에서 가져와야 하지만, 여기서는 기본값 사용)
    moe_config = {
        "n_shared_experts": 1,
        "n_routed_experts": args.num_experts,
        "n_group": 2,
        "topk_group": 2,
        "num_experts_per_tok": 2,
        "first_k_dense_replace": 8,
        "router_aux_loss_coef": 9e-1,
        "router_jitter_noise": 1e-05,
        "input_jitter_noise": 1e-05,
        "router_z_loss_coef": 1e-2,
        "ema_alpha": 0.99,
        "balancing_strength": 5e-2,
        "no_rope_layer_interval": 4,
        "use_sliding_window": True,
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 8.0
        },
        "use_bfloat16": True
    }
    
    # 모델 로드
    model, tokenizer = load_checkpoint_model(
        checkpoint_path=args.checkpoint_path,
        base_model_name=args.base_model,
        model_architecture=G3MoEForConditionalGeneration,
        moe_config=moe_config,
        device=args.device,
    )
    
    # Analyzer 초기화
    analyzer = GramSpecAnalyzer(
        num_experts=args.num_experts,
        router_dim=args.router_dim,
    )
    
    # 평가 데이터 준비
    eval_data = prepare_evaluation_data(
        tokenizer=tokenizer,
        dataset_name=args.eval_dataset,
        num_samples=args.num_samples,
        max_length=args.max_length,
        use_training_eval_set=args.use_training_eval_set,
    )
    
    # 평가 실행
    results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        eval_data=eval_data,
        analyzer=analyzer,
        device=args.device,
    )
    
    # 결과 저장
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "evaluation_results.json")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {output_file}")
    
    # 논문용 요약 출력
    if 'paper_summary' in results:
        print("\n📊 Paper Summary Metrics:")
        print(json.dumps(results['paper_summary'], indent=2))
    
    # 집계 지표 출력
    if 'aggregated_metrics' in results:
        print("\n📈 Aggregated Metrics:")
        agg = results['aggregated_metrics']
        print(f"  Load Balancing CV: {agg.get('final_load_balancing_cv', 'N/A'):.4f}")
        print(f"  Load Imbalance Ratio: {agg.get('final_load_imbalance_ratio', 'N/A'):.4f}")
        print(f"  Expert Utilization Rate: {agg.get('expert_utilization_rate', 'N/A'):.4f}")
        print(f"  Gram Orthogonality: {agg.get('avg_gram_orthogonality', 'N/A'):.4f}")


if __name__ == "__main__":
    main()

