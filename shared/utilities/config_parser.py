import argparse
import copy
import logging
import re
import yaml


def parse_config(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description='Experiment Configuration')

    parser.add_argument(
        '-c', '--config', type=str, nargs='+',
        help='A list of YAML files that provide the default values for the '
                'config. Individual fields can be overwritten through command '
                'line arguments (--set-field)'
        )
    parser.add_argument(
        '-s', '--set-field', type=str, nargs='+',
        help='Overwrite a specific config field, e.g. "-s optimizer.type=Adam".'
                'The argument will be interpreted as YAML which allows specifying '
                'lists, dicts etc and will result in a type conversion.'
        )

    config = {}
    args = parser.parse_args()

    # Load config(s), later ones overwrite earlier ones
    args.config = args.config or []
    for cfg_path in args.config:
        with open(cfg_path, 'r') as cfg_file:
            cfg = yaml.safe_load(cfg_file)
        config.update(cfg or {})

    # Modify / set config values based on command line arguments
    args.set_field = args.set_field or []
    for field_spec in args.set_field:
        # Process an argument of the type "optimizer.kwargs.lr=1.0e-3"
        try:
            name, value = field_spec.split('=')
        except Exception:
            raise ValueError(f"Unable to parse argument {field_spec}")

        nested_dict_assign(config, name, yaml.safe_load(value))

    # Process internal references in the config
    # This for example allows specifying a lr schedule in terms of training.length
    # Some time units e.g. "5.0 * EPOCHS" need to be converted after the dataset is loaded unless
    # specified directly in the config e.g. with a top level EPOCHS: 1000
    old_config = {}
    iters = 0
    while old_config != config:
        # Repeat processing to allow keys in any order (without graph processing)
        old_config = copy.deepcopy(config)
        process_references(config, flatten_dict(config))
        iters += 1

        if iters > 32:
            raise ValueError(
                f"Could not process config references. Potentially caused by circular references.\n"
                f"Current state: {config}"
            )

    return args, config


def process_references(container, conversion):
    # Update a nested container or lists and dicts in-place by:
    #   Evaluating str expressions of the form "$a OP b" where a and b are numbers or variables
    #       specified in conversion, e.g. "200 * 100 * KEY" will be replaced with 
    #       20000 * conversion[KEY].
    #       If conversion[KEY] is not numeric the return value is "20000 * {conversion[KEY]}"
    #   Replacing str expressions of the form "$KEY" with conversion[KEY]
    # Note that this is only applied to values, not keys and that spaces are required around + and -

    def process(str_expression):
        if str_expression.strip() in conversion:
            value = conversion[str_expression.strip()]
            if isinstance(value, str):
                new_value = process(value)
                if new_value[0] == '$':
                    new_value = value[1:]
                return new_value
            else:
                return copy.deepcopy(value)
        elif m := re.fullmatch(r'(.+)([\+\-\*\/%]+)(?<!\d[eE][+-])(.+)', str_expression.strip()):
            # Potential expression a OP b where a and b might be numbers or variables
            # To avoid issues with scientific notation, we don't split \d[eE][+-]
            # Insert spaces if this should be split (i.e. if a ends in [eE] and the op is [+-])
            a = process(m.group(1))
            op = process(m.group(2))
            b = process(m.group(3))
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return eval(f"{a}{op}{b}")  # eval on this should be relatively safe
            else:
                return f"{a} {op} {b}"
        else:
            # Convert to int/float if possible, otherwise return string
            try:
                value = int(str_expression)
            except Exception:
                try:
                    value = float(str_expression)
                except Exception:
                    value = str_expression
            return value

    def recurse(parent):
        enumerator = []
        if isinstance(parent, dict):
            enumerator = parent.items()
        elif isinstance(parent, list):
            enumerator = enumerate(parent)

        for key, value in enumerator:
            if isinstance(value, str) and value[0] == '$':
                new_value = process(value[1:])
                if isinstance(new_value, str):
                    new_value = '$' + new_value
                parent[key] = new_value
            else:
                recurse(value)

    recurse(container)


def nested_dict_assign(container, key, value):
    # Takes a container and key of the form 'a.b.c.d' and sets container['a']['b']['c']['d']=value
    # Will create missing intermediate values as dicts (i.e. if container['a']['b'] doesn't exist)
    key_segments = key.split('.')
    parent = container
    for key_segment in key_segments[:-1]:
        parent[key_segment] = parent.get(key_segment, {})  # Create missing intermediates
        parent = parent[key_segment]

    parent[key_segments[-1]] = value


def nested_dict_get(container, key, default=None):
    key_segments = key.split('.')
    parent = container
    for key_segment in key_segments[:-1]:
        parent = parent.get(key_segment, {})
    return parent.get(key_segments[-1], default)


def flatten_dict(container):
    def recurse(parent, prefix):
        # Assume parent is dict
        for key, value in parent.items():
            if prefix is None:
                full_key = key
            else:
                full_key = prefix + '.' + key

            if isinstance(value, dict):
                recurse(value, full_key)
            else:
                out[full_key] = value

    out = {}
    recurse(container, None)
    return out

if __name__ == '__main__':
    cfg = parse_config()
    logging.info(cfg)
