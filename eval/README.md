#### Acc 评测实现
##### 词匹配
- 通过判断答案对应的词是否在句子中出现

##### with vicuna
- 利用fastchat部署模型:替换为自己的模型路径
- 也可以通过改变端口号部署多个模型，进行投票判断
```
python3 -m fastchat.serve.controller
python3 -m fastchat.serve.model_worker --model-path /home/jiangshixin/pretrained_model/Qwen2-7B-Instruct --controller http://localhost:21001 --port 31001 --worker http://localhost:31001
/home/jiangshixin/pretrained_model/Qwen2-7B-Instruct
python3 -m fastchat.serve.openai_api_server --host localhost --port 8000
```
- 利用openai脚本进行判断:可替换模型为其他开源模型或gpt3.5等闭源api
```
openai.api_key = "EMPTY"
openai.base_url = "http://localhost:8022/v1/"

model = "Qwen2-7B-Instruct"
prompt_with_question='''
问题: "{q}"
标准答案: "{g}"
模型预测的答案: "{p}"
任务: 判断上述文本中模型预测的答案是否正确？如果完全符合回答“正确”，如果部分符合或完全不符合回答“错误”。只回答“正确”或“错误”。  
'''.format(q=question,g=ground,p=predict)
completion = openai.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user", 
            "content": prompt_with_question
        }],
    max_tokens=256
)
# print the completion
judge_result=completion.choices[0].message.content.lower()
```
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vicuna-7b-v1.5",
    "messages": [{"role": "user", "content": "你是？"}]
  }'

#### Cider(0~10) 评测实现
- PS:Cider有两种计算方法，分别对应0~1和0~10
- 使用[nlg-metricverse](https://github.com/disi-unibo-nlp/nlg-metricverse/tree/main)库
- 使用[nlg-eval](https://github.com/Maluuba/nlg-eval)库
- 在nlg-metricverse基础上引入jieba中文分词，参考[AI_challenger_Chinese_Caption](https://github.com/lxtGH/AI_challenger_Chinese_Caption/blob/master/caption_eval/coco_caption/pycxtools/coco.py)
```shell
pip install jieba
```
- 在/anaconda3/envs/XXXX/lib/python3.10/site-packages/nlgmetricverse/metrics/cider/cider_planet.py的precook()函数处更改
```python
def precook(self, s, n=4, out=False):
        """
        Takes a string as input and returns an object that can be given to
        either cook_refs or cook_test. This is optional: cook_refs and cook_test
        can take string arguments as well.
        :param s: string : sentence to be converted into ngrams
        :param n: int    : number of ngrams for which representation is calculated
        :return: term frequency vector for occuring ngrams
        """
        #THREEGOLD CHANGE
        w=jieba.cut(s.strip().replace('。',''), cut_all=False)
        s=' '.join(w)
        #THREEGOLD CHANGE
        words = s.split()
        counts = defaultdict(int)
        for k in range(1,n+1):
            for i in range(len(words)-k+1):
                ngram = tuple(words[i:i+k])
                counts[ngram] += 1
        return counts

```


##### nlg-metricverse安装及使用方法：
```
pip install nlg-metricverse

```
or   

```
git clone https://github.com/disi-unibo-nlp/nlg-metricverse.git
cd nlg-metricverse
pip install -v .
```
PS:由于 nlg-metricverse内有Bert评价函数，在第一次运行时会下载Bert相关权重，需要挂梯子或者开huggingface的镜像


使用脚本

```python
from nlgmetricverse.core import NLGMetricverse
caclute_mertrcs=["cider"]
scorer = NLGMetricverse(metrics=["cider"])
#reduce_fn在有多个参考的文本时使用。（因为caption任务一般一张图片会对应多个captions）
predict="This is a photo"
ground="This is a pig"
score = scorer( predictions=[predict], references=[ground], reduce_fn="max")
```

##### nlg-eval安装及使用方法：
```
#需要java 1.8.0
conda install openjdk 
pip install git+https://github.com/Maluuba/nlg-eval.git@master
nlg-eval --setup
```
PS:
1.由于 nlg-eval基于glove等第三方工具，会在~/.cache中下载词表等文件。若脚本速度较慢，可通过查看XXX/env/bin/nlg-eval.py中的网址手动下载并上传至~/.cache位置，然后执行nlg-eval --setup
2.参考[github_issue](https://github.com/Maluuba/nlg-eval/issues/149),由于nlg_eval依赖库gensim3.8.3不支持python3.10及以上版本，建议conda create -n nlg_eval python=3.8 -y创建一个新环境。
3.注意numpy版本numpy==1.20.3

使用脚本

```python
from nlgeval import NLGEval
nlgeval = NLGEval()  # loads the models
metrics_dict = nlgeval.compute_individual_metrics(references, hypothesis)
```