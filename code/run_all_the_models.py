import subprocess

multi_bert_models = [f"multiberts-seed_{i}" for i in range(25)]
size_models = [
    # 'mobilebert-uncased',
    'bert-base-uncased', 'bert-large-uncased']

models_to_run = size_models + multi_bert_models

for model_name in models_to_run:
    if 'large' in model_name:
        model_class = 'bert-large-uncased'
    else:
        model_class = 'bert-base-uncased'
    subprocess.run([
        "python", "eval_discriminative_models.py",
        # "--skip-intersentence",
        f"--output-file={model_name}_out.json",
        f"--intrasentence-model={model_name}",
        f"--intersentence-model={model_name}",
        '--pretrained-class=bert-base-uncased'
    ])

    subprocess.run([
        "python" , "evaluation.py",
        "--gold-file=../data/dev.json",
        f"--predictions-file=predictions/{model_name}_out.json",
        f"--output-file=output/{model_name}_icat.json",
        "--skip-intersentence=False",
    ])


only_local_bias_models = [
    'bert-base-cased', 'bert-large-cased', 'roberta-large', 'roberta-base']

for model_name in only_local_bias_models:
    subprocess.run([
        "python", "eval_discriminative_models.py",
        # roberta models don't have next sentence prediction task
        "--skip-intersentence",
        f"--output-file={model_name}_out.json",
        f"--intrasentence-model={model_name}",
        f"--intersentence-model={model_name}",
        # all of the local bias models are in pretrained class types. From what I can tell doesn't
        # actually impact anything
        f'--pretrained-class={model_name}',
    ])

    subprocess.run([
        "python" , "evaluation.py",
        "--gold-file=../data/dev.json",
        f"--predictions-file=predictions/{model_name}_out.json",
        f"--output-file=output/{model_name}_icat.json",
        "--skip-intersentence=True",
    ])
