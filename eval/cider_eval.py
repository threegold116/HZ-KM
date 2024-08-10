from nlgmetricverse.core import NLGMetricverse
# from nlgeval import compute_individual_metrics
caclute_mertrcs=["cider"]
scorer = NLGMetricverse(metrics=["cider"])
#reduce_fn在有多个参考的文本时使用。（因为caption任务一般一张图片会对应多个captions）
# predict="This is a photo"
# ground="This is a pig"
# score = scorer( predictions=[predict], references=[ground], reduce_fn="max")
# print(score)

def cider(predicts,grounds):
    if isinstance(predicts,str):
        predicts=[predicts]
    if isinstance(grounds,str):
        grounds=[grounds]
    
    score = scorer( predictions=predicts, references=grounds, reduce_fn="mean")
    # metrics_dict = compute_individual_metrics([ground],[predict])
    return score[caclute_mertrcs[0]]["score"]
print(cider(["我是梨花"],["我是李华"]))
predictions = ["Evaluating artificial text has never been so simple", "the cat is on the mat"]
references = ["Evaluating artificial text is not difficult", "The cat is playing on the mat."]
# scores = scorer(predictions, references, reduce_fn="max")
print(cider(predictions,references))