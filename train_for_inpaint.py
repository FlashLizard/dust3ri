from dust3r.training import get_args_parser
from dust3r_inpaint.training import train

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    train(args)
