from libreco.algorithms import PinSage
from libreco.data import DataInfo
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chat_models.openai import ChatOpenAI

LKPP_MODEL_PATH = "recsys_models/lkpp_model"
LKPP_MODEL_NAME = "pinsage_model_lkpp"

# tf.compat.v1.reset_default_graph()
data_info = DataInfo.load(LKPP_MODEL_PATH, model_name=LKPP_MODEL_NAME)

lkpp_recsys_model = PinSage.load(
    path=LKPP_MODEL_PATH,
    model_name=LKPP_MODEL_NAME,
    data_info=data_info,
    manual=True,
)

rec = lkpp_recsys_model.search_knn_users(1, 5)
print(rec)
print(type(rec))

class Recommendations(BaseModel):
    recs: list = Field(description="List of company_id")

model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm = model.with_structured_output(Recommendations)

def get_users_interactions():
    pass    

prompt = ""


structured_llm.invoke("Tell me a joke about cats")
    
