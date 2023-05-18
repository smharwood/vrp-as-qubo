"""
27 Jan 2023
SM Harwood

A hacky way to run unit-ish tests for vrpqubo package.
But it allows setting logging level
"""
import os
import argparse
import importlib
import logging
import re
import unittest

def main():
    """ Run unit tests """
    parser = argparse.ArgumentParser(description=
        "Run unittests in vrpqubo.tests",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("tests", nargs='+',
                        help="Name of test(s) to run, or 'all'")
    parser.add_argument("-d","--debug", action="store_true",
                        help="Log at DEBUG level")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    reg = re.compile(r"test_.*py")
    if args.tests[0] == "all":
        tests = os.listdir(os.path.join("vrpqubo","tests"))
        tests = filter(lambda fn: (reg.match(fn) is not None), tests)
    else:
        tests = args.tests

    # go through the given tests, and try to run test() method
    for test in tests:
        test = test.split('.')[0]
        try:
            test_mod = importlib.import_module(f"vrpqubo.tests.{test}")
            # this is a little hacky...
            unittest.main(test_mod, exit=False, argv=[os.path.basename(__file__)])
        except (ModuleNotFoundError, AttributeError) as error:
            print(error)

if __name__ == "__main__":
    main()
