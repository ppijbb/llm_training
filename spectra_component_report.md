# SPECTRA 컴포넌트 점검 리포트

**생성 시간**: 2026-02-02 17:27:39

**에러 타입**: `RuntimeError: The size of tensor a (0) must match the size of tensor b (2048)`

## 요약

| 항목 | 값 |
|------|-----|
| 총 테스트 | 7 |
| 성공 | 0 |
| 실패 | 7 |
| 기타 | 0 |

## 상세 결과

| 컴포넌트 | 설명 | 상태 | 소요 시간 | 에러 |
|----------|------|------|----------|------|
| `baseline_no_spectra` | SPECTRA 완전 비활성화 (Qwen3 원래 MoE) | ❌ FAILED | 1.7s | Unknown error (exit code: 1) |
| `disable_expert_dispatch` | Expert dispatch 비활성화 (hidden_states 그대로 반환) | ❌ FAILED | 1.5s | Unknown error (exit code: 1) |
| `disable_router` | SPECTRARouter 비활성화 (uniform routing) | ❌ FAILED | 1.4s | Unknown error (exit code: 1) |
| `disable_shared_experts` | shared_experts 처리 비활성화 | ❌ FAILED | 2.3s | Unknown error (exit code: 1) |
| `disable_intent_gated` | IntentGatedContextCell 비활성화 | ❌ FAILED | 3.0s | Unknown error (exit code: 1) |
| `disable_expression_proj` | ExpressionProjector 비활성화 | ❌ FAILED | 1.6s | Unknown error (exit code: 1) |
| `full_spectra` | SPECTRA 전체 활성화 (현재 상태) | ❌ FAILED | 2.6s | Unknown error (exit code: 1) |

## 분석

### ⚠️ 주의
- baseline 테스트(SPECTRA 없이 Qwen3 MoE)도 실패
- 문제가 SPECTRA가 아닌 다른 곳에 있을 수 있음

## 권장 사항

---

*이 리포트는 자동 생성되었습니다.*