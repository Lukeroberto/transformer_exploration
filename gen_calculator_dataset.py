import operator as op
import random as rand

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dp_count", type=int)
    parser.add_argument("--range", type=int, nargs="+")

    args = parser.parse_args()

    print("Number of datapoints: ", args.dp_count)
    print("Range: ", args.range)


    ops = [op.add, op.mul, op.sub, op.ifloordiv]
    ops_str = {
        op.add: "+",
        op.mul: "*",
        op.sub: "-",
        op.ifloordiv: "//"
    }

    with open('math.txt', 'w', encoding='utf-8') as f:

        for _ in range(args.dp_count):
            a = rand.randint(args.range[0], args.range[1])
            b = rand.randint(args.range[0], args.range[1])
            oper = rand.sample(ops, 1)[0]

            if oper == op.ifloordiv and b == 0:
                ans == "nan"
            else:
                ans = oper(a, b)

            dataset_line = f"{a} {ops_str[oper]} {b} = {ans}\n"
            f.write(dataset_line)

    