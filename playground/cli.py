# -*- coding: utf-8 -*-

"""Console script for playground."""
import sys
import click
import playground


@click.command()
@click.option('--name', prompt='name', help='animal number')
@click.option('--maze', prompt='maze', default='2D', help='maze name')
@click.option('--task', prompt='task', default='two cue', help='task name')
@click.option('--fpga', prompt='fpga', default='n', help='y/n')
@click.option('--decoder', prompt='decoder', default='n', help='y/n')
@click.option('--prb_file', prompt='probe file', help='probe file')
@click.option('--gui_type', prompt='raster', help='raster or feature')
def main(name, maze, task, fpga, decoder, prb_file, gui_type):
    """Console script for playground."""
    click.echo("rat #{} in {} maze doing {} task".format(name, maze, task))
    if fpga=='y':
        if decoder=='y':
            playground.run(gui_type, prb_file, BMI_ON=True, DEC_ON=True)
        else:
            playground.run(gui_type, prb_file, BMI_ON=True, DEC_ON=False)
    elif fpga=='n':
        playground.run(gui_type, prb_file, BMI_ON=False, DEC_ON=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
