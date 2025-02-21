from dotenv import load_dotenv
import os

load_dotenv()

CHROMA_PATH = "chroma-transportation-asset-pavement-management-2"
DATA_PATH = "Asset_management-1"

DEEPSEEK_API_KEY=os.environ['DeepSeek_API_Key']

MODEL_NAME="all-MiniLM-L6-v2"

# CONFIG_LIST = [{'model': 'deepseek-chat', 'api_key': DEEPSEEK_API_KEY,  'base_url' : "https://api.deepseek.com"}]

CONFIG_LIST = [{'model': 'gpt-4-turbo', 'api_key': os.environ['OPENAI_API_KEY']}]

LLM_CONFIG = {
    "temperature": 0,
    "config_list": CONFIG_LIST,
}
