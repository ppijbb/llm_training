"""
Dataset utilities for training
"""
import logging
import torch
from typing import Dict, Any, Tuple, Callable
import traceback
from PIL import Image
from data.base_model_sft_dataset import get_dataset, create_multimodal_collate_fn
from data.simple_sft_dataset import get_simple_sft_dataset, create_simple_collate_fn
from data.multi_domain_sft_dataset import get_multi_domain_sft_dataset, create_simple_collate_fn as create_multi_domain_collate_fn


class ForceImageCollatorWrapper:
    """
    Wraps a collator to ensure pixel_values (and image_grid_thw) are always present.
    If a batch is text-only, it injects a dummy black image to force the Vision Encoder 
    to execute, preventing DeepSpeed ZeRO-3 deadlocks due to divergent execution paths.
    """
    def __init__(self, collate_fn, processor, logger=None):
        self.collate_fn = collate_fn
        self.processor = processor
        self.logger = logger or logging.getLogger(__name__)
        self.dummy_pixel_values = None
        self.dummy_image_grid_thw = None
        self.dummy_created = False

    def _ensure_dummy_tensors(self):
        if self.dummy_created:
            return

        try:
            # Create a minimal dummy image (black, 28x28 to cover at least 1 patch for most models)
            dummy_image = Image.new('RGB', (28, 28), color='black')
            
            # Use processor to get tensors
            self.logger.info("🔧 ForceImageCollatorWrapper: Creating dummy image tensors...")
            
            # Check if processor supports images
            inputs = None
            if hasattr(self.processor, 'image_processor') or hasattr(self.processor, 'process_images'):
                # Call processor - handle both AutoProcessor and explicit image_processor calls if needed
                # Ideally calling processor(images=...) works for Qwen2-VL
                try:
                    inputs = self.processor(text=["dummy"], images=[dummy_image], return_tensors="pt")
                except Exception as e:
                    self.logger.warning(f"  ⚠️ Direct processor call failed: {e}. Trying image_processor directly.")
                    if hasattr(self.processor, "image_processor"):
                        inputs = self.processor.image_processor(images=[dummy_image], return_tensors="pt")
            
            if inputs is not None:
                if "pixel_values" in inputs:
                    self.dummy_pixel_values = inputs["pixel_values"]
                    self.logger.info(f"  ✅ Created dummy pixel_values with shape: {self.dummy_pixel_values.shape}")
                
                if "image_grid_thw" in inputs:
                    self.dummy_image_grid_thw = inputs["image_grid_thw"]
                    self.logger.info(f"  ✅ Created dummy image_grid_thw with shape: {self.dummy_image_grid_thw.shape}")
                
                if "pixel_values" not in inputs:
                     self.logger.warning("  ⚠️ Processor output did not contain 'pixel_values'.")
            else:
                 self.logger.warning("  ⚠️ Processor does not seem to support images, cannot create dummy tensors.")
        
        except Exception as e:
            self.logger.warning(f"  ❌ Failed to create dummy image tensors: {e}")
            self.logger.warning(traceback.format_exc())
            
        self.dummy_created = True

    def _get_dummy_image_tokens(self):
        """
        Generates the input_ids sequence corresponding to a single image.
        """
        try:
            dummy_image = Image.new('RGB', (28, 28), color='black')
            # Create a dummy conversation with just the image
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "ignore"}]}]
            
            if hasattr(self.processor, "apply_chat_template"):
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                # But we need tokens.
                inputs = self.processor(text=[text], images=[dummy_image], return_tensors="pt")
                input_ids = inputs["input_ids"][0]
                
                # We need to isolate likely image tokens.
                # Heuristic: Find the sequence of tokens that represent the image.
                # For Qwen2-VL, this is usually <|vision_start|> ... <|vision_end|>
                # However, since we just want to avoid the error, we can just grab the whole input_ids 
                # produced by the processor for this dummy image and append it? 
                # No, that includes "ignore" and system prompts.
                
                # Better approach: 
                # Just identify the specific tokens for <|vision_start|>, <|image_pad|>, <|vision_end|>
                # Qwen2-VL uses <|vision_start|> (151652), <|vision_end|> (151653), <|image_pad|> (151655)
                # We can try to look them up.
                tokenizer = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor
                
                vision_start = tokenizer.encode("<|vision_start|>", add_special_tokens=False)
                vision_end = tokenizer.encode("<|vision_end|>", add_special_tokens=False)
                image_pad = tokenizer.encode("<|image_pad|>", add_special_tokens=False)
                
                if vision_start and vision_end and image_pad:
                     start_id = vision_start[0]
                     end_id = vision_end[0]
                     
                     ids = inputs["input_ids"][0]
                     # Log what we have
                     # self.logger.info(f"  DEBUG: Generated text: {text}") 
                     # self.logger.info(f"  DEBUG: Generated ids: {ids.tolist()}")
                     # self.logger.info(f"  DEBUG: Searching for start={start_id}, end={end_id}")
                     
                     # Find start and end range
                     start_indices = (ids == start_id).nonzero(as_tuple=True)[0]
                     end_indices = (ids == end_id).nonzero(as_tuple=True)[0]
                     
                     if len(start_indices) > 0 and len(end_indices) > 0:
                         s = start_indices[0]
                         e = end_indices[0]
                         return ids[s : e + 1].clone()
                     else:
                         self.logger.warning(f"  ⚠️ Token search failed! Start indices: {start_indices}, End indices: {end_indices}")
                         self.logger.warning(f"  Generated text: {repr(text)}")
                         self.logger.warning(f"  Generated ids: {ids.tolist()}")
            
            return None
        except Exception as e:
            self.logger.warning(f"  ⚠️ Failed to generate dummy image tokens: {e}")
            self.logger.warning(traceback.format_exc())
            return None

    def __call__(self, features):
        batch = self.collate_fn(features)

        # FORCE multimodal learning: Always inject dummy image tokens
        # This ensures ALL samples follow the same multimodal computation path
        # Required for VLM training and ZeRO-3 deadlock prevention

        # Always inject dummy tokens to ensure multimodal path
        # This guarantees VLM learning regardless of dataset content
        self._ensure_dummy_tensors()

        # Inject dummy pixel_values only if missing
        pixel_values_missing = "pixel_values" not in batch or batch["pixel_values"] is None
        if pixel_values_missing and self.dummy_pixel_values is not None:
            batch["pixel_values"] = self.dummy_pixel_values.clone()

            if self.dummy_image_grid_thw is not None:
                batch["image_grid_thw"] = self.dummy_image_grid_thw.clone()

        # CRITICAL: We MUST inject image tokens into input_ids so the model processes them.
        # Otherwise Qwen2-VL throws "Image features and image tokens do not match"
        # But only if pixel_values are missing - if pixel_values exist, multimodal tokens are already there
        dummy_tokens = self._get_dummy_image_tokens()

        if dummy_tokens is not None and "input_ids" in batch and pixel_values_missing:
            dummy_len = dummy_tokens.shape[0]
            B, L = batch["input_ids"].shape

            pad_id = 0
            if hasattr(self.processor, "tokenizer") and hasattr(self.processor.tokenizer, "pad_token_id"):
                pad_id = self.processor.tokenizer.pad_token_id
            elif hasattr(self.processor, "pad_token_id"):
                pad_id = self.processor.pad_token_id

            if pad_id is None: pad_id = 0 # Safety default

            # Extend all tensors for dummy tokens
            new_input_ids = torch.full((B, L + dummy_len), pad_id, dtype=batch["input_ids"].dtype, device=batch["input_ids"].device)
            new_input_ids[:, :L] = batch["input_ids"]

            new_mask = torch.zeros((B, L + dummy_len), dtype=batch["attention_mask"].dtype, device=batch["attention_mask"].device)
            new_mask[:, :L] = batch["attention_mask"]

            new_labels = torch.full((B, L + dummy_len), -100, dtype=batch["labels"].dtype, device=batch["labels"].device)
            new_labels[:, :L] = batch["labels"]  # Copy original labels
            # Dummy tokens 부분은 이미 -100으로 초기화되어 있으므로 추가 작업 불필요

            # Replicate pixel_values for all samples if it's a single image
            if not pixel_values_missing and batch["pixel_values"] is not None:
                pixel_values = batch["pixel_values"]
                if pixel_values.dim() == 3:  # Single image [C, H, W]
                    # Replicate for batch size
                    batch["pixel_values"] = pixel_values.unsqueeze(0).expand(B, -1, -1, -1)

                if "image_grid_thw" in batch and batch["image_grid_thw"] is not None:
                    grid_thw = batch["image_grid_thw"]
                    if grid_thw.dim() == 1:  # Single image grid
                        batch["image_grid_thw"] = grid_thw.unsqueeze(0).expand(B, -1)

            # Inject into ALL samples (to ensure consistent batch processing)
            # This prevents ZeRO-3 deadlock by making all samples follow the same computation path
            for i in range(B):
                new_input_ids[i, L:] = dummy_tokens
                new_mask[i, L:] = 1  # attend to image tokens
                # labels은 이미 -100으로 초기화되어 있으므로 추가 작업 불필요

            # Update batch
            batch["input_ids"] = new_input_ids
            batch["attention_mask"] = new_mask
            batch["labels"] = new_labels

            self.logger.info(f"  🔧 Injected {dummy_len} dummy image tokens into ALL {B} samples to prevent ZeRO-3 deadlock.")
        else:
            self.logger.warning("  ⚠️ Could not generate/inject dummy tokens. Model might crash.")

        return batch

def setup_dataset(data_config: Dict[str, Any], tokenizer, logger: logging.Logger, training_config: Dict[str, Any] = None, allow_text_only: bool = False) -> Tuple[Dict, Callable]:
    """Setup training dataset"""    
    use_multi_domain = data_config.get("use_multi_domain", False)
    dataset_name = data_config.get("dataset_name", "HuggingFaceTB/smoltalk")
    max_samples = data_config.get("max_samples", 100000)
    max_samples_per_domain = data_config.get("max_samples_per_domain", None)  # multi-domain용
    max_seq_length = data_config.get("max_seq_length", 131072) or 131072
    test_size = data_config.get("test_size", 0.1)
    use_streaming = data_config.get("streaming", False)
    max_workers = data_config.get("max_workers", 4)  # multi-domain 병렬 처리용
    
    logger.info(f"Loading dataset: {dataset_name} (multi-domain: {use_multi_domain}, max_samples: {max_samples}, max_length: {max_seq_length})")
    try:
        with open("/home/conan/workspace/llm_training/sft/config/chat_template.txt", "r") as f:
            chat_template = f.read()
            if hasattr(tokenizer, 'tokenizer'):
                tokenizer.tokenizer.chat_template = chat_template
            tokenizer.chat_template = chat_template
    except Exception as e:
        logger.warning(f"Chat template setup failed: {e}")
    
    try:
        # Multi-domain 데이터셋 사용
        if use_multi_domain:
            logger.info("🔄 Using multi-domain dataset loader")
            # domain_configs가 지정되어 있으면 사용, 없으면 모든 도메인 사용
            domain_configs = data_config.get("domain_configs", None)
            
            if max_samples_per_domain is None:
                # max_samples_per_domain이 없으면 max_samples를 도메인 수로 나눔
                if domain_configs:
                    num_domains = len(domain_configs)
                else:
                    from data.multi_domain_sft_dataset import DOMAIN_DATASETS
                    num_domains = len(DOMAIN_DATASETS)
                max_samples_per_domain = max(1, max_samples // num_domains)
                logger.info(f"  Auto-calculated samples per domain: {max_samples_per_domain}")
            
            dataset = get_multi_domain_sft_dataset(
                domain_configs=domain_configs,
                tokenizer=tokenizer,
                max_length=max_seq_length,
                max_samples_per_domain=max_samples_per_domain,
                test_size=test_size,
                use_streaming=use_streaming,
                max_workers=max_workers,
                allow_text_only=allow_text_only
            )
            # Multi-domain용 collate 함수 사용 (allow_text_only=True)
            # processor 생성 (AutoProcessor 또는 tokenizer)
            # tokenizer가 이미 AutoProcessor인 경우 그대로 사용
            if hasattr(tokenizer, 'tokenizer'):
                # AutoProcessor인 경우
                processor = tokenizer
            else:
                # AutoTokenizer인 경우, tokenizer를 processor로 사용
                # (multi_domain_collate_fn이 tokenizer도 처리할 수 있음)
                processor = tokenizer
            
            collate_fn = create_multi_domain_collate_fn(processor, max_length=max_seq_length, allow_text_only=False)
            collate_fn = ForceImageCollatorWrapper(collate_fn, processor, logger)
        
        # 간단한 데이터셋 로더 사용
        elif "smoltalk" in dataset_name.lower() or "orca" in dataset_name.lower() or "llava" in dataset_name.lower():
            logger.info(f"Using simple dataset loader: {dataset_name}")
            dataset = get_simple_sft_dataset(
                dataset_name=dataset_name,
                tokenizer=tokenizer,
                max_length=max_seq_length,
                max_samples=max_samples,
                test_size=test_size,
                use_streaming=use_streaming
            )
            # 이미지 중첩 리스트 문제 해결을 위한 커스텀 data collator 사용
            # completion_only_loss 설정 (assistant_only_loss의 반대)
            completion_only_loss = False
            if training_config and "assistant_only_loss" in training_config:
                completion_only_loss = training_config["assistant_only_loss"]
            elif training_config and "completion_only_loss" in training_config:
                completion_only_loss = training_config["completion_only_loss"]

            collate_fn = create_simple_collate_fn(tokenizer, max_length=max_seq_length, completion_only_loss=completion_only_loss)
            # collate_fn = ForceImageCollatorWrapper(collate_fn, tokenizer, logger)
        else:
            # open_m_3 데이터셋 로더 시도
            dataset = get_dataset(
                tokenizer=tokenizer,
                dataset_name=data_config["dataset_name"],
                max_length=data_config["max_seq_length"],
                test_size=data_config["test_size"],
                text_only=data_config.get("text_only", False),
                streaming=data_config["streaming"]
            )
            collate_fn = create_multimodal_collate_fn(tokenizer)
            # collate_fn = ForceImageCollatorWrapper(collate_fn, tokenizer, logger)
        
        logger.info("Dataset loaded:")
        for split, data in dataset.items():
            try:
                size = data.info.dataset_size if use_streaming and hasattr(data, 'info') and hasattr(data.info, 'dataset_size') else (len(data) if hasattr(data, '__len__') else "unknown")
                logger.info(f"  {split}: {size} examples")
            except Exception as e:
                logger.warning(f"  {split}: size unknown ({e})")
        
        # 빈 데이터셋 체크
        train_dataset = dataset.get("train", None)
        if train_dataset is None:
            raise ValueError("훈련 데이터셋이 없습니다!")
        
        if use_streaming:
            if hasattr(train_dataset, 'info') and hasattr(train_dataset.info, 'dataset_size'):
                if train_dataset.info.dataset_size == 0:
                    raise ValueError("훈련 데이터셋이 비어있습니다!")
        else:
            if hasattr(train_dataset, '__len__') and len(train_dataset) == 0:
                raise ValueError("훈련 데이터셋이 비어있습니다!")

        return dataset, collate_fn
        
    except Exception as e:
        logger.error(f"❌ Dataset loading failed: {e}")
        raise RuntimeError(f"Dataset loading failed: {e}")

