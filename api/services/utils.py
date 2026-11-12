class Utils:
    
    # Use this for Display Shortening so it dosen't clip and cook the terminal output
    # Might be useful for text display in the future
    def normalize_type_name(self, value):
        if value is None:
            return ""
        return str(value).strip().lower().capitalize()

    def shorten_type_name(self, value, max_len=3):
        name = self.normalize_type_name(value)
        return name[:max_len] if name else ""
