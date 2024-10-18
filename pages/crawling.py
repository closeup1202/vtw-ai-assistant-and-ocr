import streamlit as st
import time
import asyncio
from sidebar import menu
from style import global_style
from crawling.summary import extract, convert, get_screenshot, CrawlException
from playwright.async_api import async_playwright

st.set_page_config(page_title="AI Playground - Crawling & Summary", page_icon="🤖", layout="wide")
menu()
global_style(middle_frame=True)

st.title("Crawling & Summary")

page_url = st.text_input(label="url", placeholder="ex.http://vtw.co.kr/", label_visibility="hidden")

async def progress_bar(progress):
  progress_text = "크롤링을 하고 있습니다. 잠시만 기다려주세요."
  bar = st.progress(0, text=progress_text)
  try:
    while progress[0] < 67:
      await asyncio.sleep(0.3)
      progress[0] += 3
      bar.progress(progress[0] + 1, progress_text)

    if progress[0] >= 70:
      for i in range(6):
        await asyncio.sleep(0.1) 
        bar.progress(94 + i, progress_text)
      await asyncio.sleep(0.3)
      bar.empty()
  except Exception as e:
      print(e)
      st.toast('크롤링을 할 수 없습니다. 입력하신 URL을 확인해 주세요', icon='🚨')
      bar.empty()
      return

async def crawl(url, progress):
  try:
    async with async_playwright() as p:
      browser = await p.chromium.launch()
      crawled_data = await extract(url)
      screenshot = get_screenshot(crawled_data.screenshot)
      progress[0] = 70
      await asyncio.sleep(0.5)
      result = convert(crawled_data)
      await asyncio.sleep(1)
      col_left, col_right = st.columns(2, gap="large", vertical_alignment="top")
      with col_left:
        st.image(screenshot)
      with col_right:
        show(result)
      await browser.close()
  except CrawlException as e:
    print(e)
    progress[0] = 111

@st.cache_data(show_spinner=False)
def show(result):
  st.subheader("제목")
  st.write_stream(stream_data(result["title"]))
  st.html("<hr style='margin: 2px 0' />")
  st.subheader("요약")
  st.write_stream(stream_data(result["brief_summary"]))
  st.html("<hr style='margin: 2px 0' />")
  st.subheader("상세 요약")
  st.write_stream(stream_data(result["summary"]))
  st.html("<hr style='margin: 2px 0' />")
  st.subheader("키워드")
  st.write(', '.join(result["keywords"]))

def stream_data(value):
  for word in value.split(" "):
    yield word + " "
    time.sleep(0.02)

async def main(url):
  progress = [0]
  await asyncio.gather(crawl(url, progress), progress_bar(progress))

if page_url is not None and page_url != "":
  loop = asyncio.ProactorEventLoop()
  asyncio.set_event_loop(loop)
  loop.run_until_complete(main(page_url))