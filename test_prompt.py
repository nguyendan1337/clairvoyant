import yaml
with open("config.yaml") as f: prompt = yaml.safe_load(f)["prompt"] + "hello world"
print(prompt)