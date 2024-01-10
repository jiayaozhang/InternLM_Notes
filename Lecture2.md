## 轻松玩转书生·浦语大模型趣味 Demo

1. 大模型及 InternLM 模型介绍

* 利用大量数据进行训练
* 拥有数十亿甚至数千亿个参数
* 模型在各种任务重展现出惊人的性能

2. InternLM-Chat-7B 智能对话 Demo

通过单一的代码库，InternLM支持在拥有数千个GPU的大型集群上进行预训练，并在单个GPU上进行微调，同时实现了卓越的性能优化。在1024个GPU上训练时，InternLM可以实现近90%的加速效率。
InternLM-7B包含了一个拥有70亿参数的基础模型和一个为实际场景量身定制的对话模型。该模型具有以下特点：

利用数万亿的高质量token进行训练，建立了一个强大的知识库。
支持8k token的上下文窗口长度，使得输入序列更长并增强了推理能力。

3. Lagent 智能体工具调用 Demo

Lagent介绍
Lagent是一个轻量级、开源的给予大语言模型的智能体（agent）框架，用户可以快速地将一个大语言模型转变为多种类型的智能体，并提供了一些典型工具为大语言模型赋能。

4. 浦语·灵笔图文创作理解 Demo

浦语·灵笔是基于书生·浦语大语言模型研发的视觉-语言大模型，提供出色的图文理解和创作能力，具有多项优势：

为用户打造图文并茂的专属文章。
设计了高效的训练策略，为模型注入海量的多模态概念和知识数据，赋予其强大的图文理解和对话能力。

5. 通用环境配置

pip、conda换源
pip换源设置 pip默认镜像源，升级pip到最新的版本（》=10.0.0）后进行配置，如下所示：
```
python -m pip install --upgrade pip
pip config set global.index-url https://mirrors.cernet.edu.cn/pypi/web/simple
```

conda快速换源
```
cat <<'EOF' > ~/.condarc
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF
```

HuggingFace 下载使用
HuggingFace官方提供的huggingface-cli命令行工具。
安装依赖：
```
pip install -U huggingface_hub
huggingface-cli download --resume-download internlm/internlm-chat-7b --local-dir your_path
```

模型下载
OpenXLab 可以通过指定模型仓库的地址，以及需要下载的文件的名称，文件所需下载的位置等，直接下载模型权重文件。使用python脚本下载模型首先要安装依赖，安装代码如下：pip install -U openxlab 安装完成后使用 download 函数导入模型中心的模型。
将以下代码写入Python文件，运行即可。

```
from openxlab.model import download
download(model_repo='OpenLMLab/InternLM-7b', model_name='InternLM-7b', output='your local path')
```

```
pip install modelscope
pip install transformers
```


在当前目录下载新建python文件，填入以下代码，运行即可。

```
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm-chat-7b', cache_dir='your path', revision='master')
```

