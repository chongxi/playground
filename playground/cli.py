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
def main(name, maze, task, fpga):
    """Console script for playground."""
    click.echo("rat #{} in {} maze doing {} task".format(name, maze, task))
    if fpga=='y':
        playground.run(fpga=True)
    elif fpga=='n':
        playground.run(fpga=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
