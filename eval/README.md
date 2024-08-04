#### Acc 评测实现
- 词匹配
通过判断答案对应的词是否在句子中出现
#### Cider(0~10) 评测实现
- PS:Cider有两种计算方法，分别对应0~1和0~10

- 使用[nlg-metricverse](https://github.com/disi-unibo-nlp/nlg-metricverse/tree/main)库

安装方法：
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