#!/bin/sh
# Copyright 2020 Xilinx Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

/opt/vitis_ai/conda/envs/vitis-ai-caffe/bin/yolo_detect voc/yolov3.prototxt \
                                                 voc/yolov3.caffemodel \
                                                 voc/2007_test.txt \
                                                 -out_file voc/caffe_result_fp.txt \
                                                 -confidence_threshold 0.3 \
                                                 -classes 1 \
                                                 -anchorCnt 3 \
                                                 -labels "ship" \
		                                 -model_type yolov3 \
		                                 -biases "20,23,53,34,26,88,136,34,91,91,51,166,186,80,105,203,199,173"
