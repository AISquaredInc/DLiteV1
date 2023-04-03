from utils import train, SEED
import click

@click.command()
@click.argument('local-output-dir', type = click.Path(exists = False, dir_okay = True, file_okay = False))
@click.option('--epochs', '-e', type = int, default = 3)
@click.option('--train-batch-size', type = int, default = 8)
@click.option('--eval-batch-size', type = int, default = 8)
@click.option('--lr', type = float, default = 1e-5)
@click.option('--seed', type = int, default = SEED)
@click.option('--gradient-checkpointing/--no-gradient-checkpointing', default = True)
def main(**kwargs):
    train(**kwargs)

if __name__ == '__main__':
    main()
