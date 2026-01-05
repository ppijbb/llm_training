#!/usr/bin/env python3
"""
멀티 도메인 SFT 데이터셋 테스트 스크립트
"""

import os
import sys
import logging
import traceback
from typing import Dict, Any

# Add project root directory to path for relative imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from transformers import AutoProcessor
from data.multi_domain_sft_dataset import (
    get_multi_domain_sft_dataset,
    DOMAIN_DATASETS,
    math_domain_dataset,
    science_domain_dataset,
    code_domain_dataset,
    puzzle_domain_dataset,
    vision_domain_dataset,
    ocr_domain_dataset,
    all_domains_dataset,
    log_memory_usage
)
from data.multi_domain_sft_dataset import create_simple_collate_fn

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_tokenizer_loading():
    """토크나이저 로드 테스트"""
    logger.info("=" * 60)
    logger.info("🔧 토크나이저 로드 테스트")
    logger.info("=" * 60)
    
    try:
        tokenizer = AutoProcessor.from_pretrained("google/gemma-3-4b-it")
        with open("/home/conan/workspace/llm_training/sft/config/chat_template.txt", "r") as f:
            chat_template = f.read()
        if tokenizer.tokenizer.pad_token is None:
            tokenizer.tokenizer.pad_token = tokenizer.tokenizer.eos_token
        
        logger.info(f"✅ 토크나이저 로드 성공: {tokenizer.__class__.__name__}")
        logger.info(f"   - vocab_size: {tokenizer.tokenizer.vocab_size}")
        logger.info(f"   - pad_token: {tokenizer.tokenizer.pad_token}")
        logger.info(f"   - eos_token: {tokenizer.tokenizer.eos_token}")
        
        log_memory_usage("토크나이저 로드 후")
        return tokenizer
        
    except Exception as e:
        logger.error(f"❌ 토크나이저 로드 실패: {e}")
        traceback.print_exc()
        return None


def test_single_domain_dataset(domain_name: str, tokenizer, max_samples: int = 50):
    """단일 도메인 데이터셋 테스트"""
    logger.info("=" * 60)
    logger.info(f"📦 {domain_name.upper()} 도메인 데이터셋 테스트")
    logger.info("=" * 60)
    
    domain_functions = {
        "math": math_domain_dataset,
        "science": science_domain_dataset,
        "code": code_domain_dataset,
        "puzzle": puzzle_domain_dataset,
        "vision": vision_domain_dataset,
        "ocr": ocr_domain_dataset,
        "chat": lambda tokenizer, max_samples, use_streaming: get_multi_domain_sft_dataset(
            domain_configs={"chat": DOMAIN_DATASETS["chat"]},
            tokenizer=tokenizer,
            max_samples_per_domain=max_samples,
            use_streaming=use_streaming
        ),
    }
    
    if domain_name not in domain_functions:
        logger.error(f"❌ 알 수 없는 도메인: {domain_name}")
        return None
    
    try:
        log_memory_usage(f"{domain_name} 도메인 시작")
        dataset = domain_functions[domain_name](
            tokenizer=tokenizer,
            max_samples=max_samples,
            use_streaming=True
        )
        log_memory_usage(f"{domain_name} 도메인 완료")
        
        logger.info(f"✅ {domain_name} 도메인 데이터셋 로드 성공")
        
        # 데이터셋 구조 확인
        if isinstance(dataset, dict):
            for split_name, split_data in dataset.items():
                if hasattr(split_data, '__len__'):
                    try:
                        length = len(split_data)
                        logger.info(f"   - {split_name}: {length} 샘플")
                    except:
                        logger.info(f"   - {split_name}: 스트리밍 데이터셋")
                else:
                    logger.info(f"   - {split_name}: 스트리밍 데이터셋")
        
        # 샘플 확인
        if 'train' in dataset:
            try:
                sample = dataset['train'][0]
                logger.info(f"   - 샘플 키: {list(sample.keys())}")
                
                # messages 구조 확인
                if 'messages' in sample:
                    messages = sample['messages']
                    logger.info(f"   - messages 개수: {len(messages)}")
                    if len(messages) > 0:
                        logger.info(f"   - 첫 번째 message: {messages[0]}")
                
                # images 확인
                if 'images' in sample:
                    images = sample['images']
                    logger.info(f"   - images 타입: {type(images)}")
                    if isinstance(images, list):
                        logger.info(f"   - images 개수: {len(images)}")
                
                # domain 확인
                if 'domain' in sample:
                    logger.info(f"   - domain: {sample['domain']}")
                    
            except Exception as e:
                logger.warning(f"   ⚠️ 샘플 접근 실패: {e}")
        
        return dataset
        
    except Exception as e:
        logger.error(f"❌ {domain_name} 도메인 데이터셋 로드 실패: {e}")
        traceback.print_exc()
        return None


def test_all_domains_dataset(tokenizer, max_samples_per_domain: int = 50):
    """전체 도메인 데이터셋 테스트"""
    logger.info("=" * 60)
    logger.info("📦 전체 도메인 데이터셋 테스트")
    logger.info("=" * 60)
    
    try:
        log_memory_usage("전체 도메인 시작")
        dataset = all_domains_dataset(
            tokenizer=tokenizer,
            max_samples_per_domain=max_samples_per_domain,
            use_streaming=True
        )
        log_memory_usage("전체 도메인 완료")
        
        logger.info(f"✅ 전체 도메인 데이터셋 로드 성공")
        
        # 데이터셋 구조 확인
        if isinstance(dataset, dict):
            for split_name, split_data in dataset.items():
                if hasattr(split_data, '__len__'):
                    try:
                        length = len(split_data)
                        logger.info(f"   - {split_name}: {length} 샘플")
                    except:
                        logger.info(f"   - {split_name}: 스트리밍 데이터셋")
                else:
                    logger.info(f"   - {split_name}: 스트리밍 데이터셋")
        
        # 도메인 분포 확인
        if 'train' in dataset:
            try:
                domain_counts = {}
                sample_count = min(200, len(dataset['train']) if hasattr(dataset['train'], '__len__') else 200)
                
                for i in range(sample_count):
                    try:
                        sample = dataset['train'][i]
                        domain = sample.get('domain', 'unknown')
                        domain_counts[domain] = domain_counts.get(domain, 0) + 1
                    except:
                        break
                
                logger.info(f"   - 도메인 분포 (처음 {sample_count}개 샘플):")
                for domain, count in sorted(domain_counts.items()):
                    logger.info(f"     * {domain}: {count}개")
                    
            except Exception as e:
                logger.warning(f"   ⚠️ 도메인 분포 확인 실패: {e}")
        
        return dataset
        
    except Exception as e:
        logger.error(f"❌ 전체 도메인 데이터셋 로드 실패: {e}")
        traceback.print_exc()
        return None


def test_dataset_structure(dataset, split_name: str = 'train'):
    """데이터셋 구조 검증"""
    logger.info("=" * 60)
    logger.info(f"🔍 데이터셋 구조 검증 ({split_name})")
    logger.info("=" * 60)
    
    if dataset is None or split_name not in dataset:
        logger.error(f"❌ 데이터셋 또는 split이 없습니다: {split_name}")
        return False
    
    try:
        split_data = dataset[split_name]
        
        # 샘플 개수 확인
        sample_count = 0
        try:
            if hasattr(split_data, '__len__'):
                sample_count = len(split_data)
                logger.info(f"   - 샘플 개수: {sample_count}")
            else:
                logger.info(f"   - 스트리밍 데이터셋 (길이 확인 불가)")
        except:
            logger.info(f"   - 스트리밍 데이터셋 (길이 확인 불가)")
        
        # 첫 몇 개 샘플 검증
        check_count = min(5, sample_count if sample_count > 0 else 5)
        logger.info(f"   - 검증할 샘플 수: {check_count}개")
        
        valid_samples = 0
        invalid_samples = []
        
        for i in range(check_count):
            try:
                sample = split_data[i]
                
                # 필수 필드 확인
                required_fields = ['messages']
                missing_fields = [field for field in required_fields if field not in sample]
                
                if missing_fields:
                    invalid_samples.append(f"샘플 {i}: 필수 필드 누락 - {missing_fields}")
                    continue
                
                # messages 구조 확인
                messages = sample['messages']
                if not isinstance(messages, list):
                    invalid_samples.append(f"샘플 {i}: messages가 리스트가 아님")
                    continue
                
                if len(messages) == 0:
                    invalid_samples.append(f"샘플 {i}: messages가 비어있음")
                    continue
                
                # 각 message 구조 확인
                for j, msg in enumerate(messages):
                    if not isinstance(msg, dict):
                        invalid_samples.append(f"샘플 {i}, 메시지 {j}: dict가 아님")
                        continue
                    
                    if 'role' not in msg:
                        invalid_samples.append(f"샘플 {i}, 메시지 {j}: role 필드 없음")
                        continue
                    
                    if 'content' not in msg:
                        invalid_samples.append(f"샘플 {i}, 메시지 {j}: content 필드 없음")
                        continue
                    
                    # content가 배열인지 확인
                    content = msg['content']
                    if not isinstance(content, list):
                        invalid_samples.append(f"샘플 {i}, 메시지 {j}: content가 배열이 아님 (타입: {type(content)})")
                        continue
                
                valid_samples += 1
                
            except Exception as e:
                invalid_samples.append(f"샘플 {i}: {str(e)}")
        
        logger.info(f"   - 유효한 샘플: {valid_samples}/{check_count}개")
        
        if invalid_samples:
            logger.warning(f"   - 문제가 있는 샘플:")
            for issue in invalid_samples[:10]:  # 최대 10개만 출력
                logger.warning(f"     * {issue}")
            if len(invalid_samples) > 10:
                logger.warning(f"     ... 및 {len(invalid_samples) - 10}개 더")
        
        return valid_samples == check_count
        
    except Exception as e:
        logger.error(f"❌ 데이터셋 구조 검증 실패: {e}")
        traceback.print_exc()
        return False


def test_collate_function(tokenizer, dataset, model_name: str = "google/gemma-2b-it"):
    """Collate 함수 테스트"""
    logger.info("=" * 60)
    logger.info("🔧 Collate 함수 테스트")
    logger.info("=" * 60)
    
    try:
        # Processor 생성 (multi-domain용, allow_text_only=True)
        try:
            processor = AutoProcessor.from_pretrained(model_name)
        except Exception as e:
            logger.warning(f"   ⚠️ AutoProcessor 로드 실패, tokenizer를 processor로 사용: {e}")
            # Processor가 없으면 tokenizer를 processor로 사용 (일부 모델은 tokenizer만 있음)
            processor = tokenizer
        
        collate_fn = create_simple_collate_fn(processor, max_length=2048, allow_text_only=True)
        logger.info(f"✅ Collate 함수 생성 성공 (allow_text_only=True)")
        
        if dataset is None or 'train' not in dataset:
            logger.warning("   ⚠️ 데이터셋이 없어 collate 테스트를 건너뜁니다")
            return False
        
        # 샘플 가져오기
        try:
            samples = []
            for i in range(min(2, len(dataset['train']) if hasattr(dataset['train'], '__len__') else 2)):
                try:
                    sample = dataset['train'][i]
                    samples.append(sample)
                except:
                    break
            
            if not samples:
                logger.warning("   ⚠️ 샘플을 가져올 수 없어 collate 테스트를 건너뜁니다")
                return False
            
            logger.info(f"   - 테스트할 샘플 수: {len(samples)}개")
            
            # Collate 실행
            batch = collate_fn(samples)
            logger.info(f"✅ Collate 실행 성공")
            logger.info(f"   - 배치 키: {list(batch.keys())}")
            
            for key, value in batch.items():
                if isinstance(value, (list, tuple)):
                    logger.info(f"   - {key}: {type(value).__name__} (길이: {len(value)})")
                elif hasattr(value, 'shape'):
                    logger.info(f"   - {key}: {type(value).__name__} (shape: {value.shape})")
                else:
                    logger.info(f"   - {key}: {type(value).__name__}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Collate 실행 실패: {e}")
            traceback.print_exc()
            return False
        
    except Exception as e:
        logger.error(f"❌ Collate 함수 테스트 실패: {e}")
        traceback.print_exc()
        return False


def main():
    """메인 테스트 함수"""
    logger.info("🚀 멀티 도메인 데이터셋 테스트 시작")
    log_memory_usage("프로그램 시작")
    
    # 토크나이저 로드
    tokenizer = test_tokenizer_loading()
    if tokenizer is None:
        logger.error("❌ 토크나이저 로드 실패로 테스트 중단")
        return
    
    # 각 도메인별 테스트
    test_results = {}
    for domain_name in [k for k in DOMAIN_DATASETS.keys() if k == "ocr"]:
        dataset = test_single_domain_dataset(domain_name, tokenizer, max_samples=20)
        if dataset is not None:
            test_results[domain_name] = test_dataset_structure(dataset, 'train')
        else:
            test_results[domain_name] = False
    
    # 전체 도메인 테스트
    logger.info("\n" + "=" * 60)
    logger.info("📊 전체 도메인 통합 테스트")
    logger.info("=" * 60)
    
    all_dataset = test_all_domains_dataset(tokenizer, max_samples_per_domain=20)
    if all_dataset is not None:
        structure_valid = test_dataset_structure(all_dataset, 'train')
        collate_valid = test_collate_function(tokenizer, all_dataset)
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 테스트 결과 요약")
        logger.info("=" * 60)
        
        logger.info("도메인별 테스트:")
        for domain, result in test_results.items():
            status = "✅ 성공" if result else "❌ 실패"
            logger.info(f"   - {domain}: {status}")
        
        logger.info("\n전체 도메인 테스트:")
        logger.info(f"   - 구조 검증: {'✅ 성공' if structure_valid else '❌ 실패'}")
        logger.info(f"   - Collate 함수: {'✅ 성공' if collate_valid else '❌ 실패'}")
    else:
        logger.error("❌ 전체 도메인 데이터셋 테스트 실패")
    
    log_memory_usage("테스트 완료")
    logger.info("✅ 멀티 도메인 데이터셋 테스트 완료")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⚠️ 사용자에 의해 중단됨")
    except Exception as e:
        logger.error(f"❌ 테스트 중 예외 발생: {e}")
        traceback.print_exc()
        sys.exit(1)

