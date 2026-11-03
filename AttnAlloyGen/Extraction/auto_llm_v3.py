import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from playwright.sync_api import Page, TimeoutError, sync_playwright

# ---------------------------------------------------------------------------
# LLM 配置
# ---------------------------------------------------------------------------
LLM_CONFIGS: Dict[str, Dict] = {
    "ChatGPT": {
        "url": "https://chatgpt.com/",
        "upload_method": "chooser",
        "upload_trigger_selector": "#composer-plus-btn", 
        "upload_menu_item_selector": 'div[role="menuitem"]:has-text("添加照片和文件")',
        "file_extract_wait_s": 15,
        "chat_input_selector": "#prompt-textarea",
        "send_button_selector": "button[data-testid='send-button']",
        "response_bubble_selector": "div[data-message-author-role='assistant']",
        "stop_generating_selector": "button[data-testid='stop-button']",
        "new_chat_selector": "button:has-text('新聊天'), button:has-text('New chat')",
        "login_hint_text": "Log in",
    },
    "Qwen": {
        "url": "https://chat.qwen.ai/",
        "upload_method": "chooser",
        "upload_trigger_selector": "div.mode-select-open", 
        "upload_menu_item_selector": "div.mode-select-dropdown-item:has-text('上传附件')",
        "file_extract_wait_s": 60,
        "chat_input_selector": "textarea.message-input-textarea",
        "send_button_selector": ".message-input-right-button-send .omni-button-content-btn",
        "response_bubble_selector": ".qwen-markdown-text",
        "stop_generating_selector": "", 
        "new_chat_selector": ".new-chat-button-ui",
        "login_hint_text": "登录",
    },
    "Kimi": {
        "url": "https://kimi.moonshot.cn/",
        "upload_method": "input",   
        "upload_trigger_selector": ".toolkit-trigger-btn",
        "file_input_selector": ".hidden-input",
        "chat_input_selector": ".chat-input-editor",
        "send_button_selector": ".send-button-container",
        "response_bubble_selector": ".segment-content",
        "stop_generating_selector": ".send-button-container.stop",
        "new_chat_selector": "a.new-chat-btn",
        "login_hint_text": "登录",
    },
    "Gemini": {
        "url": "https://aistudio.google.com/prompts/new_chat?model=gemini-3.1-flash-preview",
        "upload_method": "chooser",
        "upload_trigger_selector": 'button[aria-label="Add files and more"]',
        "upload_menu_item_selector": 'button:has-text("Upload files")',
        "file_extract_wait_s": 30,
        "chat_input_selector": "textarea",
        "send_button_selector": "ms-run-button button[type='submit']",
        "response_bubble_selector": ".chat-turn-container.model",
        "stop_generating_selector": "loading-indicator.generating",
        "login_hint_text": "",
},
}

# 默认 10 分钟超时 (600,000 毫秒)
DEFAULT_RESPONSE_TIMEOUT_MS = 600_000 
DEFAULT_LOGIN_TIMEOUT_SECONDS = 120
DEFAULT_UPLOAD_WAIT_MS = 15_000
POLL_INTERVAL_MS = 2_000
STABLE_SECONDS = 10         # 增加稳定性判定时间
DEBUG_SCREENSHOT_DIR = "/tmp/auto_llm_debug"


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def debug_screenshot(page: Page, tag: str, debug: bool) -> None:
    if not debug: return
    Path(DEBUG_SCREENSHOT_DIR).mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    path = f"{DEBUG_SCREENSHOT_DIR}/{ts}_{tag}.png"
    try:
        page.screenshot(path=path, full_page=False)
        print(f"  [debug] 截图已保存: {path}")
    except: pass

def clipboard_type(page: Page, text: str) -> None:
    if sys.platform == "darwin":
        subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
    elif sys.platform == "win32":
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        root.clipboard_clear()
        root.clipboard_append(text)
        root.update()
        root.after(500, root.destroy)
        root.mainloop()
    else:
        subprocess.run(["xclip", "-selection", "clipboard"], input=text.encode("utf-8"), check=True)
    modifier = "Meta" if sys.platform == "darwin" else "Control"
    page.keyboard.press(f"{modifier}+v")

# ---------------------------------------------------------------------------
# 核心业务逻辑
# ---------------------------------------------------------------------------

def upload_file(page: Page, config: Dict, file_path: Path, debug: bool, llm_name: str) -> None:
    """分发上传逻辑"""
    if config["upload_method"] == "chooser":
        trigger_sel = config["upload_trigger_selector"]
        menu_sel = config["upload_menu_item_selector"]
        
        print(f"  [{llm_name}] 正在点击菜单上传附件...")
        page.locator(trigger_sel).first.wait_for(state="visible", timeout=15_000)
        page.locator(trigger_sel).first.click()
        page.wait_for_timeout(1000) 

        with page.expect_file_chooser(timeout=10_000) as fc_info:
            page.locator(menu_sel).first.click()
        fc_info.value.set_files(str(file_path))
        
        wait_s = config.get("file_extract_wait_s", 5)
        print(f"  文件已提交，等待解析 ({wait_s}s)...")
        page.wait_for_timeout(wait_s * 1_000)
    else:
        # Input 注入模式 (Kimi)
        if config.get("upload_trigger_selector"):
            page.locator(config["upload_trigger_selector"]).first.click()
            page.wait_for_timeout(500)
        page.locator(config["file_input_selector"]).first.set_input_files(str(file_path))
        page.wait_for_timeout(3000)

def fill_prompt_and_send(page: Page, config: Dict, prompt_text: str) -> None:
    chat_input = page.locator(config["chat_input_selector"]).first
    chat_input.click()
    modifier = "Meta" if sys.platform == "darwin" else "Control"
    page.keyboard.press(f"{modifier}+a")
    page.keyboard.press("Backspace")
    clipboard_type(page, prompt_text)
    page.wait_for_timeout(1000)
    
    # 尝试寻找并点击发送按钮，若10秒内按钮不可点则强制回车
    send_sel = config["send_button_selector"]
    deadline = time.time() + 10
    while time.time() < deadline:
        btn = page.locator(send_sel).first
        if btn.count() > 0 and btn.is_visible():
            if btn.get_attribute("aria-disabled") != "true" and "disabled" not in (btn.get_attribute("class") or ""):
                btn.click()
                return
        page.wait_for_timeout(500)
    page.keyboard.press("Enter")

_GEMINI_SKIP_LINES = {'more_vert', 'thumb_up', 'thumb_down', 'content_copy','share', 'edit', 'flag', 'replay', 'stop', 'volume_up','正在思考', 'thoughts', 'chevron_right', 'Thinking','keyboard_arrow_down', 'keyboard_arrow_up'}

def get_clean_text(page: Page, selector: str, is_gemini: bool) -> str:
    locator = page.locator(selector)
    count = locator.count()
    if count == 0: return ""
    raw = locator.nth(count - 1).inner_text().strip()
    if not is_gemini: return raw
    lines = [line.strip() for line in raw.split('\n') if line.strip() and line.strip() not in _GEMINI_SKIP_LINES and not line.strip().startswith('Expand to view model') and not re.match(r'^Model \d{1,2}:\d{2}$', line.strip()) and not re.match(r'^\d+\.?\d*s$', line.strip())]
    return '\n'.join(lines)

def wait_for_response(page: Page, config: Dict, timeout_ms: int, llm_name: str) -> str:
    """核心等待逻辑：10分钟不回结果则 raise TimeoutError"""
    deadline = time.time() + (timeout_ms / 1000)
    last_text = ""
    stable_since: Optional[float] = None
    first_seen = False
    is_gemini = (llm_name == "Gemini")
    stop_sel = config.get("stop_generating_selector", "")

    while time.time() < deadline:
        text = get_clean_text(page, config["response_bubble_selector"], is_gemini)
        generating = page.locator(stop_sel).first.is_visible(timeout=200) if stop_sel else False

        if text:
            if not first_seen:
                first_seen = True
                print(f"  {llm_name} 响应中...")
            if text != last_text:
                last_text = text
                stable_since = time.time()
            else:
                if time.time() - (stable_since or time.time()) >= STABLE_SECONDS and not generating:
                    print(f"  回答稳定，提取完毕。")
                    return last_text
        page.wait_for_timeout(POLL_INTERVAL_MS)

    # 触发硬超时跳过
    raise TimeoutError(f"在 {timeout_ms/1000/60} 分钟内未能在 {llm_name} 获取到完整回答，已跳过。")

# ---------------------------------------------------------------------------
# 主控逻辑
# ---------------------------------------------------------------------------

def process_single_file(page: Page, llm_name: str, config: Dict, file_path: Path, prompt_text: str, debug: bool) -> Optional[str]:
    print(f"\n[{llm_name}] 文件: {file_path.name}")
    
    # 初始化/新建对话
    if page.url == "about:blank" or llm_name == "Gemini":
        page.goto(config["url"], wait_until="domcontentloaded", timeout=90000)
    else:
        new_btn = page.locator(config.get("new_chat_selector", "fake")).first
        if new_btn.count() > 0:
            try:
                new_btn.scroll_into_view_if_needed(timeout=5000)
                new_btn.click(force=True, timeout=10000)
            except:
                page.goto(config["url"])
            page.wait_for_timeout(1500)
        else:
            page.goto(config["url"])

    page.locator(config["chat_input_selector"]).first.wait_for(state="visible", timeout=60_000)

    upload_file(page, config, file_path, debug, llm_name)
    fill_prompt_and_send(page, config, prompt_text)
    
    # 这里会阻塞最多10分钟，超时则抛出异常给外层
    return wait_for_response(page, config, DEFAULT_RESPONSE_TIMEOUT_MS, llm_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="D:/vscode/corrosion rate/papertest")
    parser.add_argument("--prompt", default="请总结核心观点。")
    parser.add_argument("--result-dir", default="/Users/lianghongyu/Downloads/llm_results")
    parser.add_argument("--llm", default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    folder = Path(args.folder)
    files = sorted(f for f in folder.iterdir() if f.is_file() and not f.name.startswith("."))
    selected_configs = {args.llm: LLM_CONFIGS[args.llm]} if args.llm else LLM_CONFIGS

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=os.path.join(os.getcwd(), "browser_data"),
            headless=args.headless,
            viewport={"width": 1280, "height": 900},
            args=["--disable-blink-features=AutomationControlled"],
        )
        page = context.new_page()

        for llm_name, config in selected_configs.items():
            print(f"\n>>>> 正在运行模型: {llm_name}")
            save_dir = Path(args.result_dir) / llm_name
            save_dir.mkdir(parents=True, exist_ok=True)

            for file_path in files:
                res_path = save_dir / f"{file_path.name}.txt"
                if res_path.exists():
                    print(f"  [Skip] {file_path.name} 结果已存在。")
                    continue

                try:
                    # 单个文件处理
                    response = process_single_file(page, llm_name, config, file_path, args.prompt, args.debug)
                    if response:
                        res_path.write_text(response, encoding="utf-8")
                        print(f"  [Success] {file_path.name} 处理完成。")
                except TimeoutError as te:
                    print(f"  [TIMEOUT] {te}")
                    # 超时不写文件，直接处理下一个，这样续跑时会重试该文件
                except Exception as e:
                    print(f"  [ERROR] 处理 {file_path.name} 时遇到非预期错误: {e}")
                    debug_screenshot(page, "error_crash", args.debug)
                
                page.wait_for_timeout(3000)

        context.close()

if __name__ == "__main__":
    main()