# -*- coding: utf-8 -*-


import os
import argparse


if __name__ == '__main__':
    if 0:
        import solver.start_quanquan
    
        parser = argparse.ArgumentParser()
        parser.add_argument('-f', action='store', dest='flag')
        args = parser.parse_args()

        solver.start_quanquan.main(args.flag)

    if 0:
        import solver.start_densenet

        parser = argparse.ArgumentParser()
        parser.add_argument('-f', action='store', dest='flag')
        args = parser.parse_args()

        solver.start_densenet.main(args.flag)

    if 1:
        import solver.start_rpn

        parser = argparse.ArgumentParser()
        parser.add_argument('-f', action='store', dest='flag')
        args = parser.parse_args()

        solver.start_rpn.main(args.flag)
   
