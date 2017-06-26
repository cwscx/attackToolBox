from unittest import TestCase

import attackToolBox as atb

class Test(TestCase):
	def test_is_string(self):
		s = atb.nimabi()
		self.assertTrue(isinstance(s, str))