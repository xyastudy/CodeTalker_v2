#!/bin/bash
# 同步本地代码到服务器，跳过数据文件和 checkpoint
# 用法: ./sync_to_server.sh [server_alias]
# 例如: ./sync_to_server.sh droid@192.168.1.100
#       ./sync_to_server.sh gpu_server        (需要 ~/.ssh/config 里配置好别名)

SERVER="${1:-gpu_server}"   # 默认用 ~/.ssh/config 里的别名，也可以传 user@ip
REMOTE_PATH="/media/4T/xueyuan/CodeTalker_v2"

rsync -avz --progress \
  --exclude='*.pyc' \
  --exclude='__pycache__/' \
  --exclude='.git/' \
  --exclude='RUN/' \
  --exclude='checkpoint/' \
  --exclude='vocaset/vertices_npy/' \
  --exclude='vocaset/wav/' \
  --exclude='demo/output/' \
  --exclude='demo/npy/' \
  --exclude='*.mp4' \
  --exclude='*.tar.gz' \
  ./ "${SERVER}:${REMOTE_PATH}/"

echo "Sync done → ${SERVER}:${REMOTE_PATH}"
