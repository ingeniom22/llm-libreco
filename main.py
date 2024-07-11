from libreco.algorithms import PinSage
from libreco.data import DataInfo

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

print(lkpp_recsys_model.search_knn_users(1, 5))
