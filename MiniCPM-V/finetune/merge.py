from peft import PeftModel
from transformers import AutoModel,AutoTokenizer
model_type="/home/jiangshixin/pretrained_model/MiniCPM-Llama3-V-2_5" # or openbmb/MiniCPM-V-2
path_to_adapter="/home/jiangshixin/model/minicpmv/test_hz/output_minicpmv2_lora"
save_model_path="/home/jiangshixin/model/minicpmv/test_hz/minicpmv2_merge"
base_model =  AutoModel.from_pretrained(
        model_type,
        trust_remote_code=True
        )
tokenizer =AutoTokenizer.from_pretrained(
        model_type,
        trust_remote_code=True
        )
lora_model = PeftModel.from_pretrained(
    base_model,
    path_to_adapter,
    device_map="auto",
    trust_remote_code=True
)
lora_model = lora_model.merge_and_unload()
lora_model.save_pretrained(save_model_path)
tokenizer.save_pretrained(save_model_path)