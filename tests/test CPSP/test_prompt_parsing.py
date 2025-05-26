import tempfile
import os
import toml
import unittest
from src.CE.SUPS.preprocessing.parsing import  (
    get_zones_tokens,
    get_required_tokens,
    get_valid_tags,
    get_default_max_token_length,
    parse_zones_config
)

from src.CE.SUPS.preprocessing.parsing import (
    parse_block_text,
    parse_block_tags,
    parse_block_token_limit,
    extract_placeholders_from_string,
    parse_zone_resource_spec_collection,
    parse_concrete_block,
)
from src.CE.SUPS.preprocessing import TagConverter
from src.CE.SUPS.preprocessing import ResourceSpec, ZoneFactoryStub
from src.CE.SUPS.preprocessing.parsing import parse_block, parse_prompts_file, ZonesConfig


class TestZoneConfigParsing(unittest.TestCase):
    # --- get_zones_tokens tests ---
    def test_get_zones_tokens_success(self):
        cfg = {"zone_tokens": ["[A]", "[B]", "[C]"]}
        result = get_zones_tokens(cfg)
        self.assertEqual(result, ["[A]", "[B]", "[C]"])

    def test_get_zones_tokens_missing(self):
        with self.assertRaisesRegex(ValueError, "feature 'zone_tokens' for prompts toml file is missing"):
            get_zones_tokens({})

    def test_get_zones_tokens_not_list(self):
        with self.assertRaisesRegex(ValueError, "feature 'zone_tokens' for prompts toml file is not a list"):
            get_zones_tokens({"zone_tokens": "[A],[B]"})

    def test_get_zones_tokens_too_few(self):
        with self.assertRaisesRegex(ValueError, "less than two zone tokens"):
            get_zones_tokens({"zone_tokens": ["[OnlyOne]"]})

    def test_get_zones_tokens_non_str_item(self):
        with self.assertRaisesRegex(ValueError, "item in list zone_tokens is not a string"):
            get_zones_tokens({"zone_tokens": ["[A]", 123]})

    # --- get_required_tokens tests ---
    def test_get_required_tokens_none(self):
        self.assertEqual(get_required_tokens({}, ["[A]", "[B]"]), [])

    def test_get_required_tokens_not_list(self):
        with self.assertRaisesRegex(ValueError, "required_tokens.*not a list"):
            get_required_tokens({"required_tokens": "notalist"}, ["[A]"])

    def test_get_required_tokens_non_str(self):
        with self.assertRaisesRegex(ValueError, "item in list required_tokens is not a string"):
            get_required_tokens({"required_tokens": ["[A]", 123]}, ["[A]", "[B]"])

    def test_get_required_tokens_not_in_zone_tokens(self):
        with self.assertRaisesRegex(ValueError, "required token was not an option defined in zone_tokens"):
            get_required_tokens({"required_tokens": ["[X]"]}, ["[A]", "[B]"])

    def test_get_required_tokens_success(self):
        self.assertEqual(
            get_required_tokens({"required_tokens": ["[A]", "[B]"]}, ["[A]", "[B]", "[C]"]),
            ["[A]", "[B]"]
        )

    # --- get_valid_tags tests ---
    def test_get_valid_tags_missing(self):
        with self.assertRaisesRegex(ValueError, "feature 'valid_tags' for prompts toml file config does not exist"):
            get_valid_tags({})

    def test_get_valid_tags_not_list(self):
        with self.assertRaisesRegex(ValueError, "feature 'valid_tags' for prompts toml file is not a list"):
            get_valid_tags({"valid_tags": "tag1,tag2"})

    def test_get_valid_tags_non_str_item(self):
        with self.assertRaisesRegex(ValueError, "item in list valid_tags is not a string"):
            get_valid_tags({"valid_tags": ["ok", 5]})

    def test_get_valid_tags_success(self):
        tags = ["T1", "T2", "T3"]
        self.assertEqual(get_valid_tags({"valid_tags": tags}), tags)

    # --- get_default_max_token_length tests ---
    def test_get_default_max_token_length_missing(self):
        with self.assertRaisesRegex(ValueError, "feature 'default_max_token_length' for prompts toml file config does not exist"):
            get_default_max_token_length({})

    def test_get_default_max_token_length_not_int(self):
        with self.assertRaisesRegex(ValueError, "feature 'default_max_token_length' for prompts toml file is not a int"):
            get_default_max_token_length({"default_max_token_length": "100"})

    def test_get_default_max_token_length_negative(self):
        with self.assertRaisesRegex(ValueError, "feature 'default_max_token_length' for prompts toml file is not a positive int"):
            get_default_max_token_length({"default_max_token_length": 0})

    def test_get_default_max_token_length_success(self):
        self.assertEqual(get_default_max_token_length({"default_max_token_length": 1234}), 1234)

    # --- parse_zones_config tests ---
    def test_parse_zones_config_missing(self):
        with self.assertRaisesRegex(ValueError, "Config for prompts toml file is missing"):
            parse_zones_config({})

    def test_parse_zones_config_wraps_errors(self):
        bad = {"config": {"zone_tokens": "notalist"}}
        with self.assertRaisesRegex(RuntimeError, "Issue in \\[Config\\]"):
            parse_zones_config(bad)

    def test_parse_zones_config_success(self):
        toml = {
            "config": {
                "zone_tokens": ["[X]", "[Y]"],
                "required_tokens": ["[X]"],
                "valid_tags": ["A", "B"],
                "default_max_token_length": 42
            }
        }
        cfg = parse_zones_config(toml)
        self.assertIsInstance(cfg, ZonesConfig)
        self.assertEqual(cfg.zone_tokens, ["[X]", "[Y]"])
        self.assertEqual(cfg.required_tokens, ["[X]"])
        self.assertEqual(cfg.valid_tags, ["A", "B"])
        self.assertEqual(cfg.default_max_token_length, 42)


class TestConcreteBlockParser(unittest.TestCase):
    def setUp(self):
        # A minimal zones config: 3 tokens → 2 zones
        self.zone_config = ZonesConfig(
            zone_tokens=["[A]", "[B]", "[C]"],
            required_tokens=[],
            valid_tags=["T1", "T2"],
            default_max_token_length=10
        )
        # Simple tag converter: maps "T1"→0, "T2"→1
        self.tag_converter = TagConverter({"T1": 0, "T2": 1})

    # --- parse_block_text ---
    def test_parse_block_text_success(self):
        text = "[A] first [B] second [C] third"
        tokens, zones = parse_block_text({"text": text}, self.zone_config)
        self.assertEqual(tokens, ["[A]", "[B]", "[C]"])
        # zones should drop the last trailing segment
        self.assertEqual(zones, [" first ", " second "])

    def test_parse_block_text_missing(self):
        with self.assertRaisesRegex(ValueError, "feature 'text' missing in block"):
            parse_block_text({}, self.zone_config)

    def test_parse_block_text_not_string(self):
        with self.assertRaisesRegex(ValueError, "feature 'text' is not a string"):
            parse_block_text({"text": 123}, self.zone_config)

    def test_parse_block_text_no_zones(self):
        # Only one token present
        text = "[A] only one"
        with self.assertRaisesRegex(ValueError, "no zones"):
            parse_block_text({"text": text}, self.zone_config)

    def test_parse_block_text_wrong_order(self):
        # Uses tokens out of configured order
        text = "[B] bad [A] out-of-order"
        with self.assertRaisesRegex(ValueError, "wrong order, token 0"):
            parse_block_text({"text": text}, self.zone_config)

    # --- parse_block_tags ---
    def test_parse_block_tags_success(self):
        block = {"tags": [["T1"], ["T2"]]}
        tags = parse_block_tags(block, self.zone_config)
        self.assertEqual(tags, [["T1"], ["T2"]])

    def test_parse_block_tags_missing(self):
        with self.assertRaisesRegex(ValueError, "feature 'tags' missing in block"):
            parse_block_tags({}, self.zone_config)

    def test_parse_block_tags_not_list(self):
        with self.assertRaisesRegex(ValueError, "feature 'tags' is not a list"):
            parse_block_tags({"tags": "notalist"}, self.zone_config)

    def test_parse_block_tags_wrong_length(self):
        # zone_config.zone_tokens has length 3 → need 2 tag lists
        with self.assertRaisesRegex(ValueError, "does not match the number of zones"):
            parse_block_tags({"tags": [["T1"]]}, self.zone_config)

    def test_parse_block_tags_bad_item(self):
        with self.assertRaisesRegex(ValueError, "is not a list"):
            parse_block_tags({"tags": ["notalist", []]}, self.zone_config)

    def test_parse_block_tags_bad_element(self):
        with self.assertRaisesRegex(ValueError, "not a string"):
            parse_block_tags({"tags": [[123], []]}, self.zone_config)

    def test_parse_block_tags_invalid_tag(self):
        with self.assertRaisesRegex(ValueError, "is not a valid tag"):
            parse_block_tags({"tags": [["BadTag"], []]}, self.zone_config)

    # --- parse_block_token_limit ---
    def test_parse_block_token_limit_success(self):
        self.assertEqual(parse_block_token_limit({"max_gen_tokens": 5}), 5)

    def test_parse_block_token_limit_missing(self):
        with self.assertRaisesRegex(RuntimeError, "feature 'max_gen_tokens' missing in block"):
            parse_block_token_limit({})

    def test_parse_block_token_limit_not_int(self):
        with self.assertRaisesRegex(ValueError, "feature 'max_gen_tokens' is not an int"):
            parse_block_token_limit({"max_gen_tokens": "5"})

    def test_parse_block_token_limit_negative(self):
        with self.assertRaisesRegex(ValueError, "must be greater than 0"):
            parse_block_token_limit({"max_gen_tokens": 0})

    # --- extract_placeholders_from_string ---
    def test_extract_placeholders_success(self):
        s = "Hello {one}, {two}! No {three}?"
        self.assertEqual(
            extract_placeholders_from_string(s),
            ["one", "two", "three"]
        )

    def test_extract_placeholders_none(self):
        self.assertEqual(extract_placeholders_from_string("no placeholders here"), [])

    # --- parse_zone_resource_spec_collection ---
    def test_parse_zone_resource_spec_collection_success(self):
        block = {"foo": {"name": "res", "arguments": {"a": 1}}}
        specs = parse_zone_resource_spec_collection(block, "use {foo} now")
        self.assertIn("foo", specs)
        spec = specs["foo"]
        self.assertIsInstance(spec, ResourceSpec)
        self.assertEqual(spec.name, "res")
        self.assertEqual(spec.arguments, {"a": 1})

    def test_parse_zone_resource_spec_collection_missing_section(self):
        with self.assertRaisesRegex(ValueError, "feature 'foo' missing from block"):
            parse_zone_resource_spec_collection({}, "use {foo}")

    def test_parse_zone_resource_spec_collection_not_dict(self):
        block = {"foo": "notadict"}
        with self.assertRaisesRegex(ValueError, "of block not a dict"):
            parse_zone_resource_spec_collection(block, "{foo}")

    def test_parse_zone_resource_spec_collection_missing_name(self):
        block = {"foo": {"arguments": {}}}
        with self.assertRaisesRegex(ValueError, "does not have a 'name' key"):
            parse_zone_resource_spec_collection(block, "{foo}")

    def test_parse_zone_resource_spec_collection_name_not_str(self):
        block = {"foo": {"name": 123}}
        with self.assertRaisesRegex(ValueError, "not a string"):
            parse_zone_resource_spec_collection(block, "{foo}")

    def test_parse_zone_resource_spec_collection_bad_arguments(self):
        block = {"foo": {"name": "res", "arguments": "notadict"}}
        with self.assertRaisesRegex(ValueError, "must be omitted or a dictionary"):
            parse_zone_resource_spec_collection(block, "{foo}")

    def test_parse_zone_resource_spec_collection_no_arguments(self):
        block = {"foo": {"name": "res"}}
        specs = parse_zone_resource_spec_collection(block, "{foo}")
        self.assertEqual(specs["foo"].arguments, None)

    # --- parse_concrete_block ---
    def test_parse_concrete_block_success(self):
        block = {
            "text": "[A] hi [B] bye [C] end",
            "tags": [["T1"], ["T2"]],
            "max_gen_tokens": 7
        }
        stubs = parse_concrete_block(0, block, self.zone_config, self.tag_converter)
        # should produce 2 ZoneFactoryStub instances
        self.assertEqual(len(stubs), 2)
        expected_zones = [" hi ", " bye "]
        for stub, expected_token, expected_zones in zip(stubs, ["[B]", "[C]"], expected_zones):
            self.assertIsInstance(stub, ZoneFactoryStub)
            self.assertEqual(stub.prompt, expected_zones,
                             "Prompt content should match the zone body")
            self.assertEqual(stub.state_advance_token, expected_token)
            self.assertEqual(stub.resource_specs, {})
            self.assertEqual(stub.gen_token_limit, 7)

    def test_parse_concrete_block_missing_text(self):
        block = {"tags": [["T1"], ["T2"]], "max_gen_tokens": 5}
        with self.assertRaisesRegex(RuntimeError, "Issue in \\[Block\\] in toml file for block 1"):
            parse_concrete_block(1, block, self.zone_config, self.tag_converter)

    def test_parse_concrete_block_missing_tags(self):
        block = {"text": "[A] x [B] y [C] z", "max_gen_tokens": 5}
        with self.assertRaisesRegex(RuntimeError, "Issue in \\[Block\\] in toml file for block 2"):
            parse_concrete_block(2, block, self.zone_config, self.tag_converter)

    def test_parse_concrete_block_placeholder_error(self):
        block = {
            "text": "[A] {foo} [B] y [C] z",
            "tags": [[], []],
            "max_gen_tokens": 5
        }
        with self.assertRaisesRegex(RuntimeError, "Issue in \\[Block\\] in toml file for block 3"):
            parse_concrete_block(3, block, self.zone_config, self.tag_converter)

class TestMainParsingLogic(unittest.TestCase):
    def setUp(self):
        # Minimal ZonesConfig: 3 tokens -> 2 zones
        self.zone_config = ZonesConfig(
            zone_tokens=["[A]", "[B]", "[C]"],
            required_tokens=[],
            valid_tags=["T1", "T2"],
            default_max_token_length=5
        )
        self.tag_converter = TagConverter({"T1": 0, "T2": 1})

        # A simple concrete block for reuse
        self.base_block = {
            "text": "[A] x [B] y [C] z",
            "tags": [["T1"], ["T2"]],
            "max_gen_tokens": 3
        }

    # --- parse_block tests ---

    def test_parse_block_default_token_limit(self):
        block = {
            "text": "[A] a [B] b [C] c",
            "tags": [["T1"], ["T2"]]
            # no max_gen_tokens
        }
        stubs = parse_block(0, block, self.zone_config, self.tag_converter)
        # Should pick up default limit from zone_config
        for stub in stubs:
            self.assertEqual(stub.gen_token_limit, 5)

    def test_parse_block_both_repeats_and_tagset(self):
        block = self.base_block.copy()
        block["repeats"] = 2
        block["tagset"] = [[["T1"], ["T2"]]]
        with self.assertRaisesRegex(ValueError, "You cannot use both repeats and tagset; On block 1"):
            parse_block(1, block, self.zone_config, self.tag_converter)

    def test_parse_block_repeats_not_int(self):
        block = self.base_block.copy()
        block["repeats"] = "two"
        with self.assertRaisesRegex(ValueError, "'repeats' must be an int; On block 2"):
            parse_block(2, block, self.zone_config, self.tag_converter)

    def test_parse_block_repeats_less_than_one(self):
        block = self.base_block.copy()
        block["repeats"] = 0
        with self.assertRaisesRegex(ValueError, "'repeats' must be greater than 0; On block 3"):
            parse_block(3, block, self.zone_config, self.tag_converter)

    def test_parse_block_tagset_not_list(self):
        block = self.base_block.copy()
        block["tagset"] = "notalist"
        with self.assertRaisesRegex(ValueError, "'tagset' must be a list; On block 4"):
            parse_block(4, block, self.zone_config, self.tag_converter)

    def test_parse_block_tagset_empty(self):
        block = self.base_block.copy()
        block["tagset"] = []
        with self.assertRaisesRegex(ValueError, "'tagset' must have a length greater than zero; On block 5"):
            parse_block(5, block, self.zone_config, self.tag_converter)

    def test_parse_block_tagset_bad_element(self):
        block = self.base_block.copy()
        block["tagset"] = [["T1"], "notalist"]
        with self.assertRaisesRegex(ValueError, "'tagset' must only contain tag collects, got wrong type; On block 6"):
            parse_block(6, block, self.zone_config, self.tag_converter)

    def test_parse_block_repeats_success(self):
        block = self.base_block.copy()
        block["repeats"] = 2
        stubs = parse_block(7, block, self.zone_config, self.tag_converter)
        # repeats=2 → base_block expands to 2 identical blocks, each producing 2 stubs
        self.assertEqual(len(stubs), 4)
        self.assertTrue(all(isinstance(s, ZoneFactoryStub) for s in stubs))

    def test_parse_block_tagset_success(self):
        # Two tag collections
        tagset = [[["T1"], ["T2"]], [[], ["T1", "T2"]]]
        block = self.base_block.copy()
        block["tagset"] = tagset
        stubs = parse_block(8, block, self.zone_config, self.tag_converter)
        # tagset length 2 → two processed blocks → 4 stubs
        self.assertEqual(len(stubs), 4)
        # Check tags on first two stubs come from first tag collection
        self.assertEqual(stubs[0].tags.tolist(), [True, False])
        self.assertEqual(stubs[1].tags.tolist(), [False, True])
        self.assertEqual(stubs[2].tags.tolist(), [False, False])
        self.assertEqual(stubs[3].tags.tolist(), [True, True])


    # --- parse_prompts_file tests ---

    def test_parse_prompts_file_missing_blocks(self):
        content = {
            "config": {
                "zone_tokens": ["[A]", "[B]", "[C]"],
                "required_tokens": [],
                "valid_tags": ["T1", "T2"],
                "default_max_token_length": 4
            }
            # no "blocks" key
        }
        with tempfile.NamedTemporaryFile("wb", delete=False) as f:
            f.write(toml.dumps(content).encode("utf-8"))
            fname = f.name
        with self.assertRaisesRegex(ValueError, "You must specify 'blocks' in the prompts file"):
            parse_prompts_file(fname)
        os.unlink(fname)

    def test_parse_prompts_file_blocks_not_list(self):
        content = {
            "config": {
                "zone_tokens": ["[A]", "[B]", "[C]"],
                "required_tokens": [],
                "valid_tags": ["T1", "T2"],
                "default_max_token_length": 4
            },
            "blocks": "notalist"
        }
        with tempfile.NamedTemporaryFile("wb", delete=False) as f:
            f.write(toml.dumps(content).encode("utf-8"))
            fname = f.name
        with self.assertRaisesRegex(ValueError, "'blocks' must be a list"):
            parse_prompts_file(fname)
        os.unlink(fname)

    def test_parse_prompts_file_success(self):
        # Single valid block
        content = {
            "config": {
                "zone_tokens": ["[A]", "[B]", "[C]"],
                "required_tokens": [],
                "valid_tags": ["T1", "T2"],
                "default_max_token_length": 4
            },
            "blocks": [
                {
                    "text": "[A] hi [B] bye [C] end",
                    "tags": [["T1"], ["T2"]],
                    "max_gen_tokens": 2
                }
            ]
        }
        with tempfile.NamedTemporaryFile("wb", delete=False) as f:
            f.write(toml.dumps(content).encode("utf-8"))
            fname = f.name
        stubs, converter = parse_prompts_file(fname)
        self.assertIsInstance(converter, TagConverter)
        self.assertEqual(len(stubs), 2)
        self.assertTrue(all(isinstance(s, ZoneFactoryStub) for s in stubs))
        os.unlink(fname)
