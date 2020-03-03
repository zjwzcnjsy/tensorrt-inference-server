#!/bin/bash
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REPO_VERSION=${NVIDIA_TENSORRT_SERVER_VERSION}
if [ "$#" -ge 1 ]; then
    REPO_VERSION=$1
fi
if [ -z "$REPO_VERSION" ]; then
    echo -e "Repository version must be specified"
    echo -e "\n***\n*** Test Failed\n***"
    exit 1
fi

CLIENT_LOG="./client.log"
BATCHER_TEST=policy_queue_test.py
VERIFY_TIMESTAMPS=verify_timestamps.py

DATADIR=${DATADIR:="/data/inferenceserver/${REPO_VERSION}"}
OPTDIR=${OPTDIR:="/opt"}
SERVER=${OPTDIR}/tensorrtserver/bin/trtserver

source ../common/util.sh

RET=0

# Must run on a single device or else the TRTSERVER_DELAY_SCHEDULER
# can fail when the requests are distributed to multiple devices.
export CUDA_VISIBLE_DEVICES=0

# Prepare base model. Only test with custom backend as it is sufficient
rm -fr *.log *.serverlog models
cp -r ../custom_models/custom_zero_1_float32 . && \
    mkdir -p ./custom_zero_1_float32/1 && \
    cp ./libidentity.so ./custom_zero_1_float32/1/libcustom.so

(cd custom_zero_1_float32 && \
        sed -i "s/dims:.*\[.*\]/dims: \[ -1 \]/g" config.pbtxt && \
        sed -i "s/max_batch_size:.*/max_batch_size: 8/g" config.pbtxt && \
        echo "instance_group [ { kind: KIND_CPU count: 1 }]" >> config.pbtxt)

# Test max queue size
# For testing max queue size, we use delay in the custom model to
# create backlogs, "TRTSERVER_DELAY_SCHEDULER" is not desired as queue size
# is capped by max queue size.
rm -fr models && mkdir models && \
    cp -r custom_zero_1_float32 models/. && \
    (cd models/custom_zero_1_float32 && \
        echo "dynamic_batching { " >> config.pbtxt && \
        echo "    preferred_batch_size: [ 4 ]" >> config.pbtxt && \
        echo "    default_queue_policy {" >> config.pbtxt && \
        echo "        max_queue_size: 8" >> config.pbtxt && \
        echo "    }" >> config.pbtxt && \
        echo "}" >> config.pbtxt && \
        echo "parameters [" >> config.pbtxt && \
        echo "{ key: \"execute_delay_ms\"; value: { string_value: \"1000\" }}" >> config.pbtxt && \
        echo "]" >> config.pbtxt)
# Test timeout delay
# Test timeout reject
# Test batching prefered batch size after timeout policy (timing on applying timeout policy)
# Test priority levels
# Test priority levels with 1-3