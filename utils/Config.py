import os
import yaml


def update_config(conf, new_conf):
    for item in new_conf.keys():
        if type(new_conf[item]) == dict and item in conf.keys():
            conf[item] = update_config(conf[item], new_conf[item])
        else:
            conf[item] = new_conf[item]
    return conf


class Config:
    def __init__(self, path=None, config=None):
        self.is_none = False
        self.data = config if config is not None else {}
        if path is not None:
            self.path = os.path.abspath(os.path.join(os.curdir, path))
            with open(os.path.join(self.path, "default.yaml"), "rb") as default_config:
                self.data.update(yaml.load(default_config))
            for config in sorted(os.listdir(self.path)):
                if config != "defaulf.yaml" and config[-4:] in ["yaml", "yml"]:
                    with open(os.path.join(self.path, config), "rb") as config_file:
                        self.data = update_config(self.data, yaml.load(config_file))

    def set(self, key, value):
        self.data[key] = value

    def __getattr__(self, item):
        if type(self.data[item]) == dict:
            return Config(config=self.data[item])
        return self.data[item]

    def __getitem__(self, item):
        return self.data[item]
