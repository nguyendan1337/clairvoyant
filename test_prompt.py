import yaml
with open("config.yml") as f: prompt = yaml.safe_load(f)["prompt"] + "hello world"
print(prompt)