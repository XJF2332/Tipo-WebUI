## 关于模型

- 把模型放在`models`文件夹中
- **只支持**`.gguf`模型
- 点击模型列表右边的小🔄可以刷新模型列表
- 记得选择完后点击加载模型

## 关于参数

**注意：如果你修改了模型设置面板下的参数，你需要重新加载模型才能让新的参数生效**

- `max_tokens`：生成的最大长度
- `n_gpu_layers`：使用 GPU 的层数。-1 表示使用全部 GPU ，0 表示使用 CPU（模型只有几百 M 的参数量，所以其实只用 CPU
  也不见得多慢，而且还可以给 SD 省一点显存）
- `temperature`：温度。值越高，则生成结果越随机，值越低，则结果越保守
- `n_ctx`：上下文长度。这个值越大，模型能记住的信息越多，但也会导致显存占用增加
- `top_p`：这个参数决定了模型考虑的词汇范围。模型会考虑概率分布中累积概率达到 p 的所有词汇，并从中进行采样。例如，如果 top-p
  设置为 0.9 ，那么模型会在概率分布最高的 10% 的词汇中进行选择。这种方法可以增加文本生成的多样性。
- `min_p`：它设置了一个阈值，模型在生成文本时会忽略那些低于这个概率的词汇。例如，如果 min-p 设置为 0.02 ，那么任何概率低于 2%
  的词汇都不会被考虑用于文本生成。设置得太高可能会导致输出文本缺乏多样性。
- `top_k`：Top-k 采样与 top-p 类似，但是它是基于固定数量的词汇来采样的。具体来说，模型会考虑概率分布中最高的 k
  个词汇，并从中进行随机选择。如果 top-k 设置为5，那么在每一步生成文本时，模型都会从概率最高的 5
  个词汇中进行选择。这种方法可以减少低概率词汇的选取，但可能会导致一些词汇被过度使用。
- `种子`：这个不需要解释吧(≧∀≦)ゞ，设置成`-1`或点击“随机”来随机生成种子（如果你用 -1 的话，你就不知道你上次用的种子是几了）

## 关于Tags

### 模式

控制模型生成的提示词的形式

### 目标长度

控制生成的提示词的长度

### 预期质量

就是提示词里的质量标签，比如`masterpiece`这样的

### 排除标签

模型生成的提示词中，如果包含符合这些正则表达式的提示词，则符合表达式的提示词会被排除  
排除项支持正则表达式，多个排除项用英文逗号`,`分开

### 分级

控制模型生成的内容的分级

### 画师

指定你想要的画师的画风，可以留空  
不会写画师的话，可以先留空画师，然后点击`TIPO!`右边的“生成画师串”，让 AI 帮你写  
实测非 ft 模型效果并不好，大概率生成不出画师来，ft 模型可以，但生成不了画师串，只能生成一个画师

### 角色

你想要画的角色，可以留空

### Meta

其实分级也算 meta，不过这里的 meta 范围更小，像 hires 这种玄学标签就属于 meta

### Tags

你在跑图时写的其他所有标签

## 关于输出

### 结果

- 模型生成的文本
- 并不是原始结果，只是经过了处理的原始结果，不能直接用来出图，但可以获得比格式化结果更多的信息
- 如果要查看原始结果，去`GUI.py`取消注释`#for testing`的代码，然后在控制台查看原始结果

### 格式化结果

- 模型生成的文本，经过格式化处理，可以直接用来出图
- 格式化的规则会随着你选择的生成模式而改变
