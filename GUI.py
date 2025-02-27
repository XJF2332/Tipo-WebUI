import json
import os

##########################

# 加载语言
print("Loading configs...")
with open(os.path.join('Locales', 'config.json'), 'r', encoding='utf-8') as f:
    config = json.load(f)
    lang = config['language']
with open(os.path.join('Locales', f'{lang}.json'), 'r', encoding='utf-8') as f:
    locale = json.load(f)
print(locale["locale_load_success"])

##########################

# 导入库
print(locale["import_libs"])
from llama_cpp import Llama
import gradio as gr
import random
import pyperclip
import re

##########################

llm = None

##########################

# 主题
theme = gr.themes.Ocean(
    primary_hue="violet",
    secondary_hue="indigo",
    radius_size="sm",
).set(
    background_fill_primary='*neutral_50',
    border_color_accent='*neutral_50',
    color_accent_soft='*neutral_50',
    shadow_drop='none',
    shadow_drop_lg='none',
    shadow_inset='none',
    shadow_spread='none',
    shadow_spread_dark='none',
    layout_gap='*spacing_xl',
    checkbox_background_color='*primary_50',
    checkbox_background_color_focus='*primary_200'
)


##########################

# 获取模型文件列表
def list_model_files():
    models_dir = 'models'
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.gguf')]
    return [os.path.join(models_dir, f) for f in model_files]


##########################

# 随机种
def random_seed():
    return random.randint(1, 2 ** 31 - 1)


##########################

# 加载模型
def load_model(model_path, gpu, n_ctx):
    global llm
    try:
        if not model_path:
            return locale["no_model"]
        llm = None
        llm = Llama(model_path=model_path, n_gpu_layers=gpu, n_ctx=n_ctx)
        return locale["load_model_success"].format(model_path=model_path)
    except Exception as e:
        return str(e)


##########################

# 卸载模型
def unload_model():
    global llm
    llm = None
    return locale["unload_model_success"]


##########################

# 生成提示词
def gen_prompt(quality_tags, mode_tags, length_tags, tags, max_token, temp, Seed, top_p, min_p, top_k, rating,
               artist, characters, meta, length, width):
    aspect_ratio = round(length / width, 1)

    if llm is None:
        return locale["model_not_loaded"]
    else:
        if mode_tags == "None" or mode_tags == "tag_to_long" or mode_tags == "tag_to_short_to_long":
            output = llm.create_completion(
                f"quality: {quality_tags}\naspect ratio: {aspect_ratio}\ntarget: <|{length_tags}|> <|{mode_tags}|>\nrating: {rating}\nartist: {artist}\ncharacters: {characters}\nmeta: {meta}\ntag: {tags}",
                max_tokens=max_token,
                echo=True,
                temperature=temp,
                seed=Seed,
                top_p=top_p,
                min_p=min_p,
                top_k=top_k
            )
        elif mode_tags == "long_to_tag":
            output = llm.create_completion(
                f"quality: {quality_tags}\naspect ratio: {aspect_ratio}\ntarget: <|{length_tags}|> <|{mode_tags}|>\nrating: {rating}\nartist: {artist}\ncharacters: {characters}\nmeta: {meta}\nlong: {tags}",
                max_tokens=max_token,
                echo=True,
                temperature=temp,
                seed=Seed,
                top_p=top_p,
                min_p=min_p,
                top_k=top_k
            )
        else:
            output = llm.create_completion(
                f"quality: {quality_tags}\naspect ratio: {aspect_ratio}\ntarget: <|{length_tags}|> <|{mode_tags}|>\nrating: {rating}\nartist: {artist}\ncharacters: {characters}\nmeta: {meta}\nshort: {tags}",
                max_tokens=max_token,
                echo=True,
                temperature=temp,
                seed=Seed,
                top_p=top_p,
                min_p=min_p,
                top_k=top_k
            )

        return output['choices'][0]['text']


##########################

# 把 artist 放到末尾
def send_artist_to_end(text):
    pattern1 = r"\nartist:.*"
    # 移动到末尾
    text = re.sub(pattern1, "", text) + re.search(pattern1, text).group(0)
    # 去除末尾的换行
    text = text.rstrip("\n")
    return text

##########################

def gen_artist_str(prompt, max_token, temp, Seed, top_p, min_p, top_k):
    prompt = send_artist_to_end(prompt)
    output = llm.create_completion(
        prompt,
        max_tokens=max_token,
        echo=True,
        temperature=temp,
        seed=Seed,
        top_p=top_p,
        min_p=min_p,
        top_k=top_k,
        stop=["target"]
    )

    # test
    # print(output)

    return output['choices'][0]['text']

##########################

# 格式化输出
def extract_and_format(model_out, mode_tags):
    if mode_tags == "None":
        fields_to_extract = ['quality', 'artist', 'characters', 'meta', 'rating', 'tag']
    elif mode_tags == "tag_to_long":
        fields_to_extract = ['quality', 'artist', 'characters', 'meta', 'rating', 'tag', 'long']
    elif mode_tags == "tag_to_short_to_long":
        fields_to_extract = ['quality', 'artist', 'characters', 'meta', 'rating', 'tag', 'short', 'long']
    elif mode_tags == "long_to_tag":
        fields_to_extract = ['quality', 'artist', 'characters', 'meta', 'rating', 'long', 'tag']
    elif mode_tags == "short_to_long":
        fields_to_extract = ['quality', 'artist', 'characters', 'meta', 'rating', 'short', 'long']
    elif mode_tags == "short_to_tag_to_long":
        fields_to_extract = ['quality', 'artist', 'characters', 'meta', 'rating', 'short', 'tag', 'long']
    elif mode_tags == "short_to_long_to_tag":
        fields_to_extract = ['quality', 'artist', 'characters', 'meta', 'rating', 'short', 'long', 'tag']
    elif mode_tags == "short_to_tag":
        fields_to_extract = ['quality', 'artist', 'characters', 'meta', 'rating', 'short', 'tag']
    else:
        print("Error: Invalid mode_tags value")
        return "Error: Invalid mode_tags value"

    def extract_fields(model_output):
        extracted_data = {}

        for line in model_output.split('\n'):
            for field in fields_to_extract:
                if line.startswith(field + ':'):
                    extracted_data[field] = line[len(field) + 1:].strip()

        return extracted_data

    extracted_data = extract_fields(model_out)
    formatted_output = ""

    for field in fields_to_extract:
        value = extracted_data.get(field, '')
        if value:  # Only add the field if it has a value
            formatted_output += f"{value}\n\n"

    # Remove the last two newline characters to ensure no extra space at the end
    formatted_output = formatted_output.rstrip('\n')

    return formatted_output


##########################

# 排除标签
def remove_words_by_regex(sentence, pattern):
    # 移除末尾的逗号和空格（如果有的话）
    patterns = pattern.rstrip(', ')
    # 将传入的正则表达式字符串分割成列表
    pattern_list = re.split(r',\s*', patterns)
    # 使用正则表达式分割句子
    words = re.split(r',\s*', sentence)
    # 初始化一个空列表来存放过滤后的词
    filtered_words = []
    # 遍历原始单词列表
    for word in words:
        # 检查当前单词是否与任一正则表达式匹配
        should_remove = False
        for pattern in pattern_list:
            if re.match(pattern, word):
                should_remove = True
                break
        # 如果当前单词不匹配任何正则表达式，则添加到过滤后的列表中
        if not should_remove:
            filtered_words.append(word)
    # 重新组合成字符串
    result = ', '.join(filtered_words)
    return result


##########################

# 更新格式化输出
def update_format_output(formatted_text, banned_tags, mode_tags):
    text = extract_and_format(formatted_text, mode_tags)
    if banned_tags:
        formatted = remove_words_by_regex(text, banned_tags)
    else:
        formatted = text
    format_output = gr.Textbox(value=formatted, interactive=False)
    return format_output


##########################

# 复制
def copy_to_clipboard(output):
    try:
        pyperclip.copy(output)
        gr.Info(locale["copy_success"])
    except Exception as e:
        raise gr.Error(locale["copy_fail"])


##########################

# 获得模型列表
print(locale["model_searching"])
available_models = list_model_files()
print(locale["gradio_launching"])

##########################

# 刷新模型列表
def refresh_model_list(current_choice):
    new_list = list_model_files()

    if current_choice in new_list:
        new_value = current_choice
    elif new_list != []:
        new_value = new_list[0]
    else:
        new_value = None
    
    return gr.Dropdown(show_label=False, choices=new_list, scale=9, value=new_value, interactive=True)
##########################

# 加载教程
with open(os.path.join('Locales', 'Tutorials', f'{lang}.md'), "r", encoding="utf-8") as tutorial:
    tutorial_content = tutorial.read()

##########################

# gradio 界面
with gr.Blocks(theme=theme, title="TIPO") as demo:
    with gr.Row():
        with gr.Column():
            # -------------------------
            # 生成标签页
            with gr.Tab(locale["tab_generate"]):
                with gr.Row(equal_height=True):
                    # 种子
                    Seed = gr.Number(label=locale["seed"], value=-1)
                    Seed_random = gr.Button(locale["random_seed"])
                with gr.Row():
                    # 长宽比
                    img_length = gr.Number(label=locale["img_length"], value=512, minimum=256, maximum=2048, step=1)
                    img_width = gr.Number(label=locale["img_width"], value=512, minimum=256, maximum=2048, step=1)
                with gr.Row():
                    # 模式和长度标签
                    mode_tags = gr.Dropdown(
                        label=locale["mode"],
                        choices=[(locale["dropdown"]["mode_none"], "None"),
                                 (locale["dropdown"]["mode_tag2long"], "tag_to_long"),
                                 (locale["dropdown"]["mode_long2tag"], "long_to_tag"),
                                 (locale["dropdown"]["mode_short2long"], "short_to_long"),
                                 (locale["dropdown"]["mode_short2tag"], "short_to_tag"),
                                 (locale["dropdown"]["mode_tag2short2long"], "tag_to_short_to_long"),
                                 (locale["dropdown"]["mode_short2tag2long"], "short_to_tag_to_long"),
                                 (locale["dropdown"]["mode_short2long2tag"], "short_to_long_to_tag")],
                        value="None"
                    )
                    length_tags = gr.Dropdown(
                        label=locale["length"],
                        choices=[(locale["dropdown"]["length_veryshort"], "very_short"),
                                 (locale["dropdown"]["length_short"], "short"),
                                 (locale["dropdown"]["length_long"], "long"),
                                 (locale["dropdown"]["length_verylong"], "very_long")],
                        value="short"
                    )
                with gr.Row():
                    # 质量和屏蔽标签
                    quality_tags = gr.Textbox(label=locale["quality"], value="masterpiece")
                    banned_tags = gr.Textbox(label=locale["banned_tags"])
                with gr.Row():
                    # 分级和画师
                    rating_tags = gr.Dropdown(label=locale["rating"],
                                              choices=[(locale["dropdown"]["rating_safe"], "safe"),
                                                       (locale["dropdown"]["rating_sensitive"], "sensitive"),
                                                       (locale["dropdown"]["rating_nsfw"], "nsfw"),
                                                       (locale["dropdown"]["rating_explicit"], "explicit")],
                                              value="safe")
                    artist_tags = gr.Textbox(label=locale["artist"])
                with gr.Row():
                    # 角色和meta标签
                    character_tags = gr.Textbox(label=locale["character"])
                    meta_tags = gr.Textbox(label=locale["meta"], value="hires")
                # 通用标签
                tags = gr.Textbox(label=locale["general_tags"])

            # -------------------------
            # 画师标签页
            with gr.Tab(locale["tab_artist"]):
                with gr.Row(equal_height=True):
                    Seed2 = gr.Number(label=locale["seed"], value=-1)
                    Seed_random2 = gr.Button(locale["random_seed"])
                # 画师标签
                artist_tags_textbox = gr.Textbox(label=locale["artist"])

            # -------------------------
            # 设置标签页
            with gr.Tab(locale["tab_settings"]):
                # 模型设置
                gr.Markdown(locale["model_settings"])
                with gr.Row(equal_height=True):
                    model_list = gr.Dropdown(show_label=False, choices=available_models, scale=9, value=available_models[0] if available_models != [] else None)
                    refresh_model_list_btn = gr.Button("🔄", scale=1, min_width=5, interactive=True)
                with gr.Row():
                    n_ctx = gr.Number(label="n_ctx", value=2048)
                    n_gpu_layers = gr.Number(label="n_gpu_layers", value=-1)
                with gr.Row():
                    unload_btn = gr.Button(locale["model_unload"])
                    load_btn = gr.Button(locale["model_load"], variant="primary")
                load_feedback = gr.Markdown("")
                gr.Markdown(locale["generate_settings"])
                # 生成设置
                with gr.Row():
                    top_p = gr.Number(label="top_p", value=0.95)
                    min_p = gr.Number(label="min_p", value=0.05)
                with gr.Row():
                    max_tokens = gr.Number(label="max_tokens", value=1024)
                    temprature = gr.Number(label="temperature", value=0.8)
                top_k = gr.Number(label="top_k", value=60)

            # -------------------------
            # 教程标签页
            with gr.Tab(locale["tab_tutorial"]):
                gr.Markdown(tutorial_content)

        # -------------------------
        # 结果展示
        with gr.Column():
            with gr.Row():
                upsampling_btn = gr.Button("TIPO！", variant="primary", scale=2)
                copy_btn = gr.Button(locale["copy_to_clipboard"], scale=1)
                gen_artists = gr.Button(locale["generate_artists"], scale=1)
            with gr.Row():
                raw_output = gr.Textbox(label=locale["result"], interactive=False)
                formatted_output = gr.Textbox(label=locale["formatted_result"], interactive=False)
                # 更新格式化输出
                raw_output.change(update_format_output, inputs=[raw_output, banned_tags, mode_tags],
                                  outputs=formatted_output)
                artist_tags_textbox.change(update_format_output, inputs=[artist_tags_textbox, banned_tags, mode_tags],
                                           outputs=formatted_output)

    # -------------------------
    # 写提示词
    upsampling_btn.click(
        fn=gen_prompt,
        inputs=[quality_tags, mode_tags, length_tags, tags, max_tokens, temprature, Seed, top_p, min_p, top_k,
                rating_tags, artist_tags, character_tags, meta_tags, img_length, img_width],
        outputs=raw_output
    )

    # -------------------------
    # 加载模型
    load_btn.click(
        fn=load_model,
        inputs=[model_list, n_gpu_layers, n_ctx],
        outputs=load_feedback
    )

    # -------------------------
    # 卸载模型
    unload_btn.click(
        fn=unload_model,
        inputs=None,
        outputs=load_feedback
    )

    # -------------------------
    # 随机种子
    Seed_random.click(
        fn=random_seed,
        inputs=None,
        outputs=Seed
    )
    Seed_random2.click(
        fn=random_seed,
        inputs=None,
        outputs=Seed2
    )

    # -------------------------
    # 复制到剪贴板
    copy_btn.click(
        fn=copy_to_clipboard,
        inputs=formatted_output,
        outputs=None
    )

    # -------------------------
    # 生成画师串
    gen_artists.click(
        fn=gen_artist_str,
        inputs=[raw_output, max_tokens, temprature, Seed2, top_p, min_p, top_k],
        outputs=artist_tags_textbox
    )

    # -------------------------
    # 刷新模型列表
    refresh_model_list_btn.click(
        fn=refresh_model_list,
        inputs=model_list,
        outputs=model_list
    )


demo.launch()
