"""TOML boolean documentation and diagnostics must match strict value types."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('generate_ecm_config',ROOT/'tools/gen/generate_ecm_config.py')
gen=importlib.util.module_from_spec(spec);spec.loader.exec_module(gen)


class BooleanMetadata(unittest.TestCase):
    def test_all_boolean_domains_use_toml_literals(self):
        for option in gen.load_schema()['options']:
            if option['type']=='bool':
                self.assertEqual(option['domain'],'[false|true]',option['key'])

    def test_internal_integer_parser_does_not_change_documented_default_type(self):
        option=next(o for o in gen.load_schema()['options'] if o['key']=='stage2_giant_tail_balance')
        self.assertIn('default=false',gen.annotation(option))

    def test_generator_rejects_a_misleading_boolean_domain(self):
        schema=copy.deepcopy(gen.load_schema())
        schema['options']=[o for o in schema['options'] if not o.get('_mirror')]
        next(o for o in schema['options'] if o['key']=='stage2_giant_tail_balance')['domain']='[0|1]'
        with patch.object(gen.json,'loads',return_value=schema):
            with self.assertRaisesRegex(ValueError,'boolean.*domain'):
                gen.load_schema()


if __name__=='__main__':unittest.main()
