import argparse

def qargs(*args):
    parser = argparse.ArgumentParser()
    for arg_name, help_text in args:
        parser.add_argument(arg_name, help=help_text)
    return parser.parse_args()
