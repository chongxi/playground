# -*- coding: utf-8 -*-

"""Console script for playground."""
import sys
import click
import playground


@click.command()
# @click.option('--name', prompt='name', help='animal number')
# @click.option('--maze', prompt='maze', default='2D', help='maze name')
# @click.option('--task', prompt='task', default='two cue', help='task name')
# @click.option('--gui_type', prompt='raster', help='raster or feature')
def main():
    """Console script for playground."""
    playground.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
