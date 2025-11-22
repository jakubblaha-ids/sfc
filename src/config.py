import json
import os


class ConfigManager:
    """Manages application configuration stored in a JSON file"""

    def __init__(self, config_dir="tmp", config_filename="sfc_config.json"):
        """
        Initialize the configuration manager.

        Args:
            config_dir: Directory where config file is stored, relative to cwd (default: tmp)
            config_filename: Name of the config file (default: sfc_config.json)
        """
        # Make config_dir relative to current working directory
        self.config_dir = os.path.join(os.getcwd(), config_dir)
        self.config_filename = config_filename
        self.config_path = os.path.join(self.config_dir, config_filename)
        self.config = self._load_config()

    def _load_config(self):
        """Load configuration from file, or return defaults if file doesn't exist"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading config: {e}. Using defaults.")
                return self._get_default_config()
        else:
            return self._get_default_config()

    def _get_default_config(self):
        """Return default configuration"""
        return {
            "last_map_path": None
        }

    def save_config(self):
        """Save current configuration to file"""
        try:
            # Ensure config directory exists
            os.makedirs(self.config_dir, exist_ok=True)

            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except IOError as e:
            print(f"Error saving config: {e}")

    def get_last_map_path(self):
        """Get the path to the last imported/exported map"""
        return self.config.get("last_map_path")

    def set_last_map_path(self, path):
        """
        Set the path to the last imported/exported map.

        Args:
            path: Full path to the map file
        """
        self.config["last_map_path"] = path
        self.save_config()

    def get(self, key, default=None):
        """
        Get a configuration value.

        Args:
            key: Configuration key
            default: Default value if key doesn't exist

        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)

    def set(self, key, value):
        """
        Set a configuration value and save.

        Args:
            key: Configuration key
            value: Value to set
        """
        self.config[key] = value
        self.save_config()
