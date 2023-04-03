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
@click.option('--cuda/--no-cuda', default = True)
def main(local_output_dir, epochs, train_batch_size, eval_batch_size, lr, seed, gradient_checkpointing, cuda):
    train(
        local_output_dir = local_output_dir,
        epochs = epochs,
        train_batch_size = train_batch_size,
        eval_batch_size = eval_batch_size,
        lr = lr,
        seed = seed,
        gradient_checkpointing = gradient_checkpointing,
        cuda = cuda
    )

if __name__ == '__main__':
    main()
