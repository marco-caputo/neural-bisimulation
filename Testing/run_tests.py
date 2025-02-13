import unittest

loader = unittest.TestLoader()
suite = loader.discover('Testing', pattern='*_test.py')

runner = unittest.TextTestRunner()
runner.run(suite)