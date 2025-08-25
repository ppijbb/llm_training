#!/bin/bash

if ! command -v uv &> /dev/null; then
    echo "uv 명령어를 찾을 수 없습니다. 먼저 uv를 설치하세요."
    exit 1
fi

# (필요하다면) 가상환경 활성화
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

uv run python data/upload_sft_dataset.py \
    merge \
        --output_name=open_m_3 \
        --num_workers=16 \
        --local_path /mnt/disks/local-ssd/ \
        --max_samples 100

uv run python data/upload_sft_dataset.py \
    upload \
        --dataset_path /mnt/disks/local-ssd/open_m_3 \
        --repo_id Gunulhona/t_test \
        --num_workers 16 \
        --chunk_size 10000 \
        --single_repo 