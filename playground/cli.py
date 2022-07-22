# -*- coding: utf-8 -*-

"""Console script for playground."""
import sys
import click
import playground


@click.command()
@click.option('--bmi_update_rule', prompt='moving_average/fixed_length', help='moving_average', default='moving_average')
@click.option('--posterior_threshold', prompt='posterior_threshold', help='[0.01]', default=0.01)
@click.option('--bmi_mode', prompt='bmi_mode', help='True/False', default=True)
# @click.option('--maze', prompt='maze', default='2D', help='maze name')
# @click.option('--task', prompt='task', default='two cue', help='task name')
# @click.option('--gui_type', prompt='raster', help='raster or feature')
def main(bmi_update_rule, posterior_threshold, bmi_mode):
    """Console script for playground."""
    print(f'bmi_mode:{bmi_mode}')
    playground.run_experiment(bmi_update_rule, posterior_threshold, bmi_mode=bmi_mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
