def get_datestring():
    from datetime import datetime

    now = datetime.now()
    return f"{now.year:04}{now.month:02}{now.day:02}-{now.hour:02}{now.minute:02}{now.second:02}-{now.microsecond:06}"


class Logger:
    def __init__(self, base_path, echo=True):
        from os.path import exists

        assert exists(base_path)
        self.enabled = True
        self.echo = echo
        self.timestamp = get_datestring()
        self.filename = f"{base_path}/{self.timestamp}.txt"
        self.final_filename = f"{base_path}/{self.timestamp}.log"
        self.final_log = f""
        with open(self.filename, "w+") as fhandle:
            fhandle.write("")

    def set_header(self, a="", b="", c="", d=""):
        self.final_log += f"{a:20},{b:20},{c:20},{d:20},"

    def add_data(self, *datum):
        for data in datum:
            self.final_log += f"{data:.4f},"

    def finalize(self):
        if self.enabled:
            with open(self.final_filename, "w+") as fhandle:
                fhandle.write(self.final_log + "\n")

    def add(self, formatted_text):
        if isinstance(formatted_text, dict):
            import json

            return self.add(json.dumps(formatted_text, indent=4))
        if isinstance(formatted_text, tuple):
            import json

            return self.add(json.dumps(formatted_text, indent=4))
        if self.enabled:
            with open(self.filename, "a") as fhandle:
                if self.echo:
                    print(formatted_text, end="")
                fhandle.write(formatted_text)
        return self

    def addnl(self, formatted_text):
        self.add(formatted_text)
        if self.enabled:
            with open(self.filename, "a") as fhandle:
                fhandle.write("\n")
            if self.echo:
                print()
        return self
