# 轻松玩转书生·浦语大模型趣味Demo

# 1、SSH配置

为了方便后续使用，首先介绍一下我的SSH配置。我用本地的vscode连接远程开发机，下面是简单的操作流程。

1、首先我们在VSCode中安装SSH插件，首先打开插件市场,然后搜索Remote-SSH,点击安装。![1704379727316](image/Chapter2/1704379727316.png)

![1704380356899](image/Chapter2/1704380356899.png)

2、然后我们在命令行中(如果你的VSCode下面没有命令行窗口，可以按快捷键 `Ctrl+~`，就会出现啦)，生成 SSH 密钥对，在命令行中输入：

```powershell
ssh-keygen -t rsa
```

之后一直按回车就行，就可以生成对应的密钥文件，并保存在以下位置。我们用记事本打开这个 `id_rsa.pub`文件，然后复制里面的内容。

![1704380533097](image/Chapter2/1704380533097.png)

3、进入**InternStudio 控制台配置SSH公钥：**

首先打开访问管理·

![1704380786411](image/Chapter2/1704380786411.png)

然后添加SSH公钥，将刚才复制的 `.pub`文件中的内容复制在：

![1704380937387](image/Chapter2/1704380937387.png)

4、连接服务器：

- 首先进入控制台并按照教程启动开发机，然后点击SSH连接，并复制里面的内容：

![1704380996809](image/Chapter2/1704380996809.png)

![1704381069841](image/Chapter2/1704381069841.png)

- 打开VSCode 并配置SSH，点击以下按键，添加配置，之后会弹出一个对话框，将刚才复制的命令填进去然后按回车确认，并选择

![1704381192794](image/Chapter2/1704381192794.png)![1704381266394](image/Chapter2/1704381266394.png)

![1704381306785](image/Chapter2/1704381306785.png)

- 进行远程连接：在VSCode界面按下 `Shift+Ctrl+P`并输入 `remote-ssh`，然后就会显示我们刚才添加的主机，直接点击就可以连上啦!

![1704381476186](image/Chapter2/1704381476186.png)

![1704381546274](image/Chapter2/1704381546274.png)

![1704381559554](image/Chapter2/1704381559554.png)

- 注意：释放开发机后重新创建可能会SSH连接不上，重新复制开发机那里的命令，导入到Remote-SSH配置里就可以了

  ![1704641989638](image/Chapter2/1704641989638.png)

# 2、大模型及InternLM模型简介

# 3、InternLM-Chat-7B 智能对话 Demo

1、按照教程，首先克隆一个新的虚拟环境，为了避免各种包的版本问题，建议大家在个人开发过程中，每个项目分别创建对应的虚拟环境。

```
/root/share/install_conda_env_internlm_base.sh internlm-demo
```

然后再激活环境

`conda activate internlm-demo`

安装依赖：

```
# 升级pip
python -m pip install --upgrade pip

pip install modelscope==1.9.5
pip install transformers==4.35.2
pip install streamlit==1.24.0
pip install sentencepiece==0.1.99
pip install accelerate==0.24.1
```

2、下载模型

这里[InternStudio](https://studio.intern-ai.org.cn/)平台 `share目录`中准备了现成的模型，直接复制即可，避免长时间下载：

```shell
mkdir -p /root/model/Shanghai_AI_Laboratory
cp -r /root/share/temp/model_repos/internlm-chat-7b /root/model/Shanghai_AI_Laboratory
```

然后 `clone` 代码，在 `/root` 路径下新建 `code` 目录，然后切换路径, clone 代码.

```shell
cd /root/code
git clone https://gitee.com/internlm/InternLM.git
```

切换 commit 版本，与教程 commit 版本保持一致，可以让大家更好的复现。

```shell
cd InternLM
git checkout 3028f07cb79e5b1d7342f4ad8d11efad3fd13d17
```

后面按照教程来就行

# 问题汇总

- HuggingFace下载连接超时问题
  修改源解决问题：

  ```shell
  export HF_ENDPOINT=https://hf-mirror.com
  ```
  默认情况下，huggingface_hub下载的文件将被下载到由HF_HOME环境变量定义的缓存目录中（如果未指定，则为**~/.cache/huggingface/hub** ）
