import torch
from transformers import AutoTokenizer, AutoModel,AutoConfig
import os

#cuda是否存在
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")
#torch版本
print(torch.__version__)

torch.cuda.empty_cache()
# 模型路径
# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上上层目录
parent_dir = os.path.dirname(current_dir)
# 构建模型路径
model_path = os.path.join(parent_dir, "chatglm26b")
print(os.path.exists(model_path))
#加载模型

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()

model = model.eval()
print(f"加载模型完成")
response, history = model.chat(tokenizer, "你能干什么", history=[])
print(response)
