import asyncio
import json
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from playwright.async_api import Page, BrowserContext
from google.auth.transport.requests import Request
import google.auth

def get_access_token_method():
    """Get access token using google.auth.default()"""
    credentials, project = google.auth.default(
        scopes=['https://www.googleapis.com/auth/cloud-platform']
    )
    credentials.refresh(Request())
    return credentials.token

async def main():
    print("🔗 Hooks Example: Demonstrating recommended usage")

    # 1) Configure the browser
    browser_config = BrowserConfig(
        headless=True,
        verbose=True
    )

    # 2) Configure the crawler run
    crawler_run_config = CrawlerRunConfig(
        js_code="window.scrollTo(0, document.body.scrollHeight);",
        wait_for="body",
        cache_mode=CacheMode.BYPASS
    )

    # 3) Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)

    #
    # Define Hook Functions
    #

    async def on_browser_created(browser, **kwargs):
        # Called once the browser instance is created (but no pages or contexts yet)
        print("[HOOK] on_browser_created - Browser created successfully!")
        # Typically, do minimal setup here if needed
        return browser

    async def on_page_context_created(page: Page, context: BrowserContext, **kwargs):
        # Called right after a new page + context are created (ideal for auth or route config).
        print("[HOOK] on_page_context_created - Setting up page & context.")

        # Example 1: Route filtering (e.g., block images)
        # async def route_filter(route):
        #     if route.request.resource_type == "image":
        #         print(f"[HOOK] Blocking image request: {route.request.url}")
        #         await route.abort()
        #     else:
        #         await route.continue_()

        # await context.route("**", route_filter)
        access_token = get_access_token_method()
        await context.set_extra_http_headers({
            "DATALAB_TUNNEL_TOKEN": "TOKEN"
        })
        print(access_token)
        print("[HOOK] on_page_context_created - GCP Auth header is set.")
        # Example 2: (Optional) Simulate a login scenario
        # (We do NOT create or close pages here, just do quick steps if needed)
        # e.g., await page.goto("https://example.com/login")
        # e.g., await page.fill("input[name='username']", "testuser")
        # e.g., await page.fill("input[name='password']", "password123")
        # e.g., await page.click("button[type='submit']")
        # e.g., await page.wait_for_selector("#welcome")
        # e.g., await context.add_cookies([...])
        # Then continue

        # Example 3: Adjust the viewport
        await page.set_viewport_size({"width": 1080, "height": 600})
        return page

    async def before_goto(
        page: Page, context: BrowserContext, url: str, **kwargs
    ):
        # Called before navigating to each URL.
        print(f"[HOOK] before_goto - About to navigate: {url}")
        # e.g., inject custom headers
        # await page.set_extra_http_headers({
        #     "Custom-Header": "my-value"
        # })
        return page

    async def after_goto(
        page: Page, context: BrowserContext, 
        url: str, response, **kwargs
    ):
        # Called after navigation completes.
        print(f"[HOOK] after_goto - Successfully loaded: {url}")
        # e.g., wait for a certain element if we want to verify
        try:
            await page.wait_for_selector('.content', timeout=1000)
            print("[HOOK] Found .content element!")
        except:
            print("[HOOK] .content not found, continuing anyway.")
        return page

    async def on_user_agent_updated(
        page: Page, context: BrowserContext, 
        user_agent: str, **kwargs
    ):
        # Called whenever the user agent updates.
        print(f"[HOOK] on_user_agent_updated - New user agent: {user_agent}")
        return page

    async def on_execution_started(page: Page, context: BrowserContext, **kwargs):
        # Called after custom JavaScript execution begins.
        print("[HOOK] on_execution_started - JS code is running!")
        return page

    async def before_retrieve_html(page: Page, context: BrowserContext, **kwargs):
        # Called before final HTML retrieval.
        print("[HOOK] before_retrieve_html - We can do final actions")
        # Example: Scroll again
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
        return page

    async def before_return_html(
        page: Page, context: BrowserContext, html: str, **kwargs
    ):
        # Called just before returning the HTML in the result.
        print(f"[HOOK] before_return_html - HTML length: {len(html)}")
        return page

    #
    # Attach Hooks
    #

    crawler.crawler_strategy.set_hook("on_browser_created", on_browser_created)
    crawler.crawler_strategy.set_hook(
        "on_page_context_created", on_page_context_created
    )
    crawler.crawler_strategy.set_hook("before_goto", before_goto)
    crawler.crawler_strategy.set_hook("after_goto", after_goto)
    crawler.crawler_strategy.set_hook(
        "on_user_agent_updated", on_user_agent_updated
    )
    crawler.crawler_strategy.set_hook(
        "on_execution_started", on_execution_started
    )
    crawler.crawler_strategy.set_hook(
        "before_retrieve_html", before_retrieve_html
    )
    crawler.crawler_strategy.set_hook(
        "before_return_html", before_return_html
    )

    await crawler.start()

    # 4) Run the crawler on an example page
    url = "https://pqbmrig4hndx5dwgt43tb3pq7a-dot-europe-west4.dataproc.googleusercontent.com/gateway/default/sparkhistory/history/app-20250219140302-0000/jobs/?user.name=anonymous"
    result = await crawler.arun(url, config=crawler_run_config)

    if result.success:
        print("\nCrawled URL:", result.url)
        print("HTML length:", len(result.html))
    else:
        print("Error:", result.error_message)

    await crawler.close()

if __name__ == "__main__":
    asyncio.run(main())