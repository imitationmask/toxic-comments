from himl import ConfigProcessor


def create_config(model):
    config_processor = ConfigProcessor()
    path = "./config_gen/" + model #+ "/hyper_params.yaml"
    filters = () # can choose to output only specific keys
    exclude_keys = () # can choose to remove specific keys
    output_format = "yaml" # yaml/json

    return config_processor.process(path=path, filters=filters, exclude_keys=exclude_keys,
                                    output_format=output_format, print_data=True)