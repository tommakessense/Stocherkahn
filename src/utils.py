import argparse

def imp_args(func):

    def define_args():
        parser = argparse.ArgumentParser()
        func(parser)
        return parser.parse_args()

    return define_args

