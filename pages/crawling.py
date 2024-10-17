import streamlit as st
import asyncio
import time
from sidebar import menu
from style import global_style
from crawling.summary import summarize
from playwright.async_api import async_playwright

st.set_page_config(page_title="AI Playground - Crawling & Summary", page_icon="🤖", layout="wide")
menu()
global_style(middle_frame=True)

st.title("Crawling & Summary")

page_url = st.text_input("URL", placeholder="ex.http://vtw.co.kr/")

async def progress_bar(progress):
  progress_text = "크롤링을 하고 있습니다. 잠시만 기다려주세요."
  bar = st.progress(0, text=progress_text)
  while progress[0] < 100:
    await asyncio.sleep(0.2) 
    if progress[0] < 100:
      progress[0] = progress[0] + 3    
      bar.progress(progress[0] + 1, progress_text)
    else:
      break;
  bar.progress(100, progress_text)
  await asyncio.sleep(0.5) 
  bar.empty()

async def crawl(url, progress):
  async with async_playwright() as p:
    browser = await p.chromium.launch()
    result = await summarize(url)
    progress[0] = 100
    await asyncio.sleep(0.5)
    st.write("summary:", result["title"])
    await browser.close()

async def main(url):
  progress = [0]
  await asyncio.gather(crawl(url, progress), progress_bar(progress))

if page_url is not None and page_url != "":
  loop = asyncio.ProactorEventLoop()
  asyncio.set_event_loop(loop)
  loop.run_until_complete(main(page_url))