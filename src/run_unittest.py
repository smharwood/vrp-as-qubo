"""
27 Jan 2023
SM Harwood

Run unit-ish tests for vrpqubo package
"""
import os
import argparse
import importlib
import logging
import re

logging.basicConfig(level=logging.DEBUG)

def main():
    """ Run unit tests """
    parser = argparse.ArgumentParser(description=
        "Run unittests in vrpqubo.tests",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("tests", nargs='+',
                        help="Name of test(s) to run, or 'all'")
    args = parser.parse_args()

    reg = re.compile(r"test_.*py")
    if args.tests[0] == "all":
        tests = os.listdir(os.path.join("vrpqubo","tests"))
        tests = filter(lambda fn: (reg.match(fn) is not None), tests)
    else:
        tests = args.tests

    # go through the given tests, and try to run test() method
    for test in tests:
        test = test.split('.')[0]
        print("\n")
        try:
            test_mod = importlib.import_module(f"vrpqubo.tests.{test}")
            test_mod.test()
        except (ModuleNotFoundError, AttributeError) as error:
            print(error)

if __name__ == "__main__":
    main()
