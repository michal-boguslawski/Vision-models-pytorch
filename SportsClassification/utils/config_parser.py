import os
import yaml
from utils.filesystem import cache_message


def yaml_include(loader, node):
    filename = os.path.join(os.path.dirname(loader.name), node.value)
    with open(filename, 'r') as f:
        return yaml.safe_load(f)

yaml.SafeLoader.add_constructor("!include", yaml_include)


class ConfigParser:
    def __init__(self, config_path: str = "configs/default.yaml"):
        self.config_path = config_path
        raw_config = self._load_yaml(config_path)
        self.config = self._sanitize(raw_config)

    def _load_yaml(self, path: str) -> dict:
        """Load a YAML file."""
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def get(self, key, default=None):
        """Access nested keys safely."""
        return self.config.get(key, default)

    def merge(self, other_config: dict):
        """Merge another config dictionary into current config (override defaults)."""
        self._deep_update(self.config, other_config)

    @staticmethod
    def _deep_update(d: dict, u: dict):
        """Recursively update dict d with values from u."""
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = ConfigParser._deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def __getitem__(self, key):
        return self.config[key]

    def __repr__(self):
        return yaml.dump(self.config, sort_keys=False)

    def _sanitize(self, d):
        """Recursively convert scientific string numbers like '1e-4' to floats."""
        if isinstance(d, dict):
            return {k: self._sanitize(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [self._sanitize(v) for v in d]
        elif isinstance(d, str):
            try:
                return float(d) if any(ch in d for ch in ['e', '.']) else d
            except ValueError:
                return d
        return d

    def save(self, path: str):
        with open(path, "w") as f:
            yaml.dump(self.config, f, sort_keys=False, default_flow_style=False)
        cache_message(f"Config saved to {path}")
