import yaml


class MyDumper(yaml.Dumper):
    """
    indent properly when dumping to a .yaml config file.
    https://stackoverflow.com/questions/25108581/python-yaml-dump-bad-indentation
    """
    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)