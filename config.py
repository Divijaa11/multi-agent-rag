from dotenv import load_dotenv
import os

load_dotenv()

CHROMA_PATH = "chroma-transportation-asset-pavement-management"
DATA_PATH = "Asset_management"

DEEPSEEK_API_KEY=os.environ['DeepSeek_API_Key']

MODEL_NAME="all-MiniLM-L6-v2"

CONFIG_LIST = [{'model': 'deepseek-chat', 'api_key': DEEPSEEK_API_KEY,  'base_url' : "https://api.deepseek.com"}]

LLM_CONFIG = {
    "temperature": 0,
    "config_list": CONFIG_LIST,
}
