# -*- coding: utf-8 -*-

"""Console script for playground."""
import sys
import click
import playground


@click.command()
@click.option('--name', prompt='name', help='animal number')
@click.option('--maze', prompt='maze', default='2D', help='maze name')
@click.option('--task', prompt='task', default='two cue', help='task name')
def main(name, maze, task):
    """Console script for playground."""
    playground.main()
    click.echo("rat #{} in {} maze doing {} task".format(name, maze, task))
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
