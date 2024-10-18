import os
import json
import streamlit as st
import base64
from PIL import Image
from io import BytesIO
from crawl4ai import AsyncWebCrawler
from pydantic import BaseModel, Field
from crawl4ai.chunking_strategy import RegexChunking
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from googletrans import Translator

class PageSummary(BaseModel):
  title: str = Field(..., description="Title of the page.")
  summary: str = Field(..., description="Summary of the page.")
  brief_summary: str = Field(..., description="Brief summary of the page.")
  keywords: list = Field(..., description="Keywords assigned to the page.")

class CrawlException(Exception):
  def __init__(self, msg):
    super().__init__(msg)

extraction_strategy = LLMExtractionStrategy(
  provider="openai/gpt-4o-mini",
  api_token=os.getenv('OPENAI_API_KEY'),
  schema=PageSummary.model_json_schema(),
  extraction_type="schema",
  apply_chunking=False,
  instruction=(
    "From the crawled content, extract the following details: "
    "1. Title of the page "
    "2. Summary of the page, which is a detailed summary "
    "3. Brief summary of the page, which is a paragraph text "
    "4. Keywords assigned to the page, which is a list of keywords. "
    'The extracted JSON format should look like this: '
    '{ "title": "Page Title", "summary": "Detailed summary of the page.", '
    '"brief_summary": "Brief summary in a paragraph.", "keywords": ["keyword1", "keyword2", "keyword3"] }'
  )
)

async def extract(url):
  async with AsyncWebCrawler(verbose=True) as crawler:
    result = await crawler.arun(
      url=url,
      word_count_threshold=1,
      extraction_strategy=extraction_strategy,
      chunking_strategy=RegexChunking(),
      screenshot=True,
    )
    if result.success:
      return result
    else:
      raise CrawlException(result.error_message)

@st.cache_data(show_spinner=False)
def translate(value):
  result = Translator().translate(str(value), src='en', dest='ko')
  return result.text

def convert(result):
  encoded_result = result.extracted_content.encode('utf-8', errors='ignore').decode("unicode_escape")
  result_json = json.loads(encoded_result)
  result_json_dict = result_json[0]
  converted_keywords = [ translate(ele) for ele in result_json_dict['keywords']]
  converted_dict = { ele: translate(value) for ele, value in result_json_dict.items()}
  converted_dict["keywords"] = converted_keywords
  return converted_dict

def get_screenshot(screenshot):
  image_data = base64.b64decode(screenshot)
  return Image.open(BytesIO(image_data))