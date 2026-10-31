"""Exact result identities above Python's default4300 decimal-digit limit."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ecm_report'))
from result_format import modulus


class LargeIntegerResult(unittest.TestCase):
    def test_supported_wide_target_does_not_depend_on_process_digit_limit(self):
        previous=sys.get_int_max_str_digits()
        try:
            sys.set_int_max_str_digits(4300)
            self.assertEqual(modulus({'N':'1'+'0'*4932}),10**4932)
            self.assertEqual(modulus({'N':'  +1'+'0'*4932+'  '}),10**4932)
        finally:sys.set_int_max_str_digits(previous)

    def test_invalid_wide_decimal_is_rejected(self):
        for suffix in ('x','_0','.0','e2'):
            with self.assertRaises(ValueError):modulus({'N':'1'+'0'*4932+suffix})

    def test_small_integer_behavior(self):
        self.assertEqual(modulus({'N':'123456789'}),123456789)
        self.assertEqual(modulus({'N':123456789}),123456789)

    def test_large_known_factor_is_checked_exactly(self):
        self.assertEqual(modulus({'k':1,'b':10,'n':4932,'c':-1,'known-factors':['1'*4932]}),9)
        with self.assertRaises(ValueError):
            modulus({'k':1,'b':10,'n':4932,'c':-1,'known-factors':['1'*4931+'2']})


if __name__=='__main__':unittest.main()
