import unittest

loader = unittest.TestLoader()
suite = loader.discover('testing', pattern='*_test.py')

runner = unittest.TextTestRunner()
runner.run(suite)