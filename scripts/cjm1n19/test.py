import evaluate

metric = evaluate.load('glue', 'mnli')
print(metric)
