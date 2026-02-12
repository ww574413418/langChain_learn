from playwright.sync_api import sync_playwright

def run():
    # 使用playwright上下文管理器
    with sync_playwright() as p:
        browser = p.webkit.launch(headless=False)
        # 创建一个新页面
        page = browser.new_page()
        # 导航到指定url
        page.goto("https://baidu.com/")

        # 获取并打印标题
        title = page.title()
        print(f"the title of page is:{title}")

        browser.close()

if __name__ == '__main__':
    run()
