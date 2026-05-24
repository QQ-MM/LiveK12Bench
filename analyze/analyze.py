"""Parse OCR'd exam papers (markdown / jsonl) into structured JSON.

The parser asks an LLM (multi-turn dialogue) to extract per-question
fields defined in `analyze/configs/chinese_k12_exam.py`, then optionally copies
referenced images to the output directory.

Usage:
    python analyze/analyze.py \\
        --ocr-dir /path/to/ocr_results \\
        --save-dir /path/to/parsed_json \\
        [--model gpt-5]
"""

import os
import sys
import json
import shutil
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluate.util.llm import call_llm
from configs.chinese_k12_exam import target_items, paper_template

system_prompt = (
    "你是一名细致且不会犯错的文件整理助手，根据我提供的文件文本，"
    "提取我需要的信息并输出为JSON格式。\n\n"
    "文件中包含多道题目，整理每个对象的信息并放在list(dict)结构中，"
    "每一个对象的dict中要包含的键值及其要求如下：{target_items}。"
    "对于找不到信息的键返回null值。\n\n"
    "为了更方便定位上述目标信息，你收到的信息很可能符合以下格式特点：{paper_template}。"
    "如果文件不符合上述格式，灵活根据其内容分析并完成任务。\n\n"
    "##注意##\n"
    "1. 不要对文件内容（包括文本内容和latex公式内容）进行任何修改，仅摘取整理对应的信息即可。\n"
    "2. 确保解析文件中所有题目，不要遗漏。\n"
    "3. 仅输出可以直接被解析为JSON的结果，确保该结果解析时不会报错。\n"
    "4. 如果已经没有内容可以提取，仅输出【完成】。\n"
)


def analyze(file_path, save_dir, model="gpt-5"):
    """Parse a single OCR file and write per-question JSON to save_dir."""
    saved_path = os.path.join(save_dir, os.path.splitext(os.path.basename(file_path))[0] + ".json")
    print(f"Analyzing {file_path}")
    print(f"Save_path: {saved_path}")

    if os.path.exists(saved_path):
        print("Saved path already exists, skipping it.")
        return None

    with_img = False
    root_path = None
    if ".md" not in file_path and ".jsonl" not in file_path:
        # MinerU-style output: a directory containing auto/ subfolder with images/ and the .md
        file_path += "/auto"
        assert os.path.exists(os.path.join(file_path, "images")), f"{file_path} 图片目录缺失！"
        with_img = True
        root_path = file_path
        file_path = os.path.join(file_path, os.path.basename(file_path.replace("/auto", "")) + ".md")

    with open(file_path, "r", encoding="utf-8") as file:
        if ".jsonl" in file_path:
            for line in file:
                ocr_text = json.loads(line)["text"]
                break
        elif ".md" in file_path:
            ocr_text = file.read()

    prompt = [{"type": "text", "text": ocr_text}]
    messages = [
        {"role": "system", "content": system_prompt.format(target_items=target_items, paper_template=paper_template)},
        {"role": "user", "content": prompt},
    ]
    history_conv = messages

    response = ""
    saved_json = []
    while "【完成】" not in response:
        response, _ = call_llm(messages=history_conv, model=model)
        response = response.replace("\\\\", "\\").replace("\\", "\\\\").replace('\\\\"', "'")
        print(response)
        if "【完成】" in response:
            break
        history_conv.append({"role": "assistant", "content": response})
        history_conv.append({"role": "user", "content": "继续"})
        try:
            saved_json += json.loads(response)
        except Exception as e:
            print(response)
            print(e)

    # Copy referenced images into save_dir/images.
    if with_img:
        os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
        delete_items = []
        for item in saved_json:
            img_paths = item.get("图像", "")
            if img_paths:
                for img_path in img_paths:
                    src = os.path.join(root_path, img_path)
                    if os.path.exists(src):
                        shutil.copy(src, os.path.join(save_dir, "images", os.path.basename(img_path)))
                    else:
                        delete_items.append(item)
        for item in delete_items:
            saved_json.remove(item)

    with open(saved_path, "w", encoding="utf-8") as file:
        json.dump(saved_json, file, ensure_ascii=False, indent=4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocr-dir", required=True,
                        help="Directory containing OCR'd exam papers (.md / .jsonl).")
    parser.add_argument("--save-dir", required=True,
                        help="Where to write parsed per-paper JSON files.")
    parser.add_argument("--model", default="gpt-5",
                        help="LLM used to extract structured fields.")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    for file in os.listdir(args.ocr_dir):
        analyze(os.path.join(args.ocr_dir, file), args.save_dir, model=args.model)


if __name__ == "__main__":
    main()
