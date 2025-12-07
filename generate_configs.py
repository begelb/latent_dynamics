import os
import ast
import pprint

EX = 1

base_config_path = f'config/Leslie.txt'

output_dir_name = f'configs/prelim_{EX}'
if not os.path.exists(output_dir_name):
    os.makedirs(output_dir_name)

summary = []
config_index = 0

subdiv_range = range(18, 23)

# for k in subdiv_range:
for ex_index in range(0, 20):
    with open(base_config_path, 'r') as f:
        base_config_str = f.read()
        base_config = ast.literal_eval(base_config_str)

    new_config = base_config.copy()

    new_config['ex_index'] = ex_index

    new_config['output_dir'] = f'output/Leslie/{ex_index}'
    new_config['model_dir'] = f'output/Leslie/{ex_index}/models'
    new_config['log_dir'] = f'output/Leslie/{ex_index}/logs'

    config_filename = f'config_{config_index}.txt'
    config_filepath = os.path.join(output_dir_name, config_filename)

    with open(config_filepath, 'w') as f:
        f.write(pprint.pformat(new_config))

    # Add the details to our summary list
    summary.append(f'{config_filename}: ex_index={ex_index}')

    config_index += 1

summary_filename = output_dir_name + '/config_summary.txt'
with open(summary_filename, 'w') as f:
    f.write('\n'.join(summary))

print(f"Successfully generated {config_index} configuration files in '{output_dir_name}'.")
print(f"A summary has been saved to '{summary_filename}'.")
