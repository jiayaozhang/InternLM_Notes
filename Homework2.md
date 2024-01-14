# 4个Demo轻松玩转书生·浦语大模型

## 准备

1. 在实操前我们需要对部署大模型应用有一个大概的了解，首先我们需要有算力，这里以InternStudio为例，如果您的账号没有算力请报名书生·浦语大模型实战营第二期。如下图，笔者已经在 InternStudio 平台创建了开发机并配置了 SSH 连接。


2. 其次，我们需要创建 conda 环境，并安装 python 依赖：

```bash
bash # 请每次使用 jupyter lab 打开终端时务必先执行 bash 命令进入 bash 中
conda create --name internlm-demo --clone=/root/share/conda_envs/internlm-base # 创建conda环境
conda activate internlm-demo # 激活conda环境

# 升级pip
python -m pip install --upgrade pip

pip install modelscope==1.9.5
pip install transformers==4.35.2
pip install streamlit==1.24.0
pip install sentencepiece==0.1.99
pip install accelerate==0.24.1
pip install transformers==4.33.1 timm==0.4.12 sentencepiece==0.1.99 gradio==3.44.4 markdown2==2.4.10 xlsxwriter==3.1.2 einops accelerate

```

接着我们还需要下载模型,开发机里提供了一些预先下载好的模型，我们只需拷贝即可使用：

```bash
mkdir -p /root/model/Shanghai_AI_Laboratory
cp -r /root/share/temp/model_repos/internlm-chat-7b /root/model/Shanghai_AI_Laboratory

cp -r /root/share/temp/model_repos/internlm-xcomposer-7b /root/model/Shanghai_AI_Laboratory

```

当然也可以使用HuggingFace、ModelScope 或者OpenXLab 下载，如MOdelScope下载internlm-chat-7b:


```bash
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm-chat-7b', cache_dir='/root/model', revision='v1.0.3')
```


1. InternLM-Chat智能对话Demo

通过这个Demo，我们通过 steamlit 的方式运行基于 InternLM-Chat-7B 模型的智能对话机器人。

代码准备：

```bash
cd /root/code
git clone https://gitee.com/internlm/InternLM.git
```

修改 /root/code/InternLM/web_demo.py 中模型的路径，替换为本地路径，接着运行 Demo：

```bash
streamlit run web_demo.py --server.address 127.0.0.1 --server.port 6006
```


![300字故事.png](images/300字故事.png)

* Familiar with hugging face function, use `huggingface_hub python` to download `InternLM-20B` and `config.json`

![huggingface_download.png](images/huggingface_download.png)


2. Lagent智能体工具调用Demo

通过书生·浦语大模型开源开发全链路体系中应用(智能体)Lagent 实现能解数学题的Demo

代码准备：
```bash
cd /root/code
git clone https://gitee.com/internlm/lagent.git
cd /root/code/lagent
pip install -e . # 源码安装Lagent
```

修改 /root/code/lagent/examples/react_web_demo.py，主要是替换 init_model方法中模型路径，action_list移除GoogleSearch() 接着运行命令：

```bash
streamlit run /root/code/lagent/examples/react_web_demo.py --server.address 127.0.0.1 --server.port 6006
```

![lagent.png](images/lagent.png)

3. 浦语·灵笔图文理解创作 Demo 1

通过internlm-xcomposer-7b 模型部署一个图文理解创作 Demo ，能够理解图片信息、能够创作图文长篇文章等。

代码准备：


```bash
cd /root/code
git clone https://gitee.com/internlm/InternLM-XComposer.git
```

代码运行：

```bash
cd /root/code/InternLM-XComposer
python examples/web_demo.py  \
    --folder /root/model/Shanghai_AI_Laboratory/internlm-xcomposer-7b \
    --num_gpus 1 \
    --port 6006
```

![mutigrid.png](images/mutigrid.png)

4. 浦语·灵笔图文理解创作 Demo 2

![ai-article.png](images/ai-article.png)

![ai-article2.png](images/ai-article2.png)

参考下面生成的文章
> [又见法兰克福](images/io.md)