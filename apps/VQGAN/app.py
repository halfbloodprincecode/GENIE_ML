from libs.basicTime import getTimeHR
from apps.VQGAN.modules.args import parser
# from pytorch_lightning.trainer import Trainer


def main():
    now = getTimeHR(now=True, split='T', dateFormat='%Y-%m-%d', timeFormat='%H-%M-%S')
    opt, unknown = parser()
    

if __name__ == "__main__":
    main()