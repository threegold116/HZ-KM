from nlgmetricverse.core import NLGMetricverse
caclute_mertrcs=["cider"]
scorer = NLGMetricverse(metrics=["cider"])
#reduce_fn在有多个参考的文本时使用。（因为caption任务一般一张图片会对应多个captions）
# predict="This is a photo"
# ground="This is a pig"
# score = scorer( predictions=[predict], references=[ground], reduce_fn="max")
# print(score)

def cider(predict,ground):
    score = scorer( predictions=[predict], references=[ground], reduce_fn="max")
    return score[caclute_mertrcs[0]]["score"]