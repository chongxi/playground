from playground.base.jovian import Jovian
from playground.utils import Timer
import sys
import click
import time


@click.command()
@click.option('--n', prompt='# lines', default=10,   help='number of lines', type=int)
@click.option('--v', prompt='verbose', default=True, help='number of lines', type=bool)
def main(n, v):
    jov = Jovian()
    for _ in range(n):
        with Timer('', verbose=v): 
            print(jov.get())


if __name__ == '__main__':
    sys.exit(main()) 
