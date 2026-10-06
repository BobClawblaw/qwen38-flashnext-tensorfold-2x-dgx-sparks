#!/bin/bash
export HOME=/opt/qwen38-flashnext-tensorfold
export PATH=/opt/qwen38-flashnext-tensorfold/.local/bin:/usr/local/bin:/usr/bin:/bin
export PROFILE=concurrent
cd /opt/qwen38-flashnext-tensorfold
exec ./run.sh
